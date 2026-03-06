import argparse
import copy
import json
from tqdm import tqdm
import torch
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import (
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *


def resolve_model_path(model_path):
    """Resolve model path: if it's a local directory, validate it exists;
    if it's a HF repo ID, try to find it in local cache to avoid network issues."""
    if os.path.isdir(model_path):
        return model_path
    # Try to resolve from HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_path:
                # Return the latest snapshot path
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    snapshot_path = str(revisions[0].snapshot_path)
                    print(f"[INFO] Found cached model: {snapshot_path}")
                    return snapshot_path
    except Exception:
        pass
    return model_path


SCHEDULER_MAP = {
    'dpm': DPMSolverMultistepScheduler,
    'ddim': DDIMScheduler,
    'euler_a': EulerAncestralDiscreteScheduler,
    'euler': EulerDiscreteScheduler,
    'pndm': PNDMScheduler,
    'lms': LMSDiscreteScheduler,
}


def get_scheduler(scheduler_name, model_path):
    if scheduler_name not in SCHEDULER_MAP:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {list(SCHEDULER_MAP.keys())}")
    return SCHEDULER_MAP[scheduler_name].from_pretrained(model_path, subfolder='scheduler')


def get_latent_shape(pipe):
    """Get latent space dimensions from the pipeline."""
    channels = pipe.unet.config.in_channels
    vae_scale = pipe.vae_scale_factor
    return channels, vae_scale


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.model_path = resolve_model_path(args.model_path)
    scheduler = get_scheduler(args.scheduler, args.model_path)

    # Always create a DDIM scheduler for inversion (DDIM inversion requires DDIM math)
    ddim_scheduler_for_inversion = DDIMScheduler.from_pretrained(
        args.model_path, subfolder='scheduler'
    )

    from_pretrained_kwargs = dict(
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    if args.revision:
        from_pretrained_kwargs['revision'] = args.revision
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            **from_pretrained_kwargs,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # Store the generation scheduler and the DDIM inversion scheduler
    generation_scheduler = pipe.scheduler
    inversion_scheduler = ddim_scheduler_for_inversion

    # Load LoRA weights if specified (for failure case testing)
    if args.lora_path is not None:
        print(f"[INFO] Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)

    # For ControlNet mode, load ControlNet model
    controlnet = None
    if args.mode == 'controlnet' and args.controlnet_path is not None:
        from diffusers import ControlNetModel
        print(f"[INFO] Loading ControlNet from {args.controlnet_path}")
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_path, torch_dtype=torch.float16
        ).to(device)

    # Determine latent space dimensions
    latent_channels = pipe.unet.config.in_channels
    vae_scale = pipe.vae_scale_factor
    latent_h = args.image_length // vae_scale
    latent_w = (args.image_width if args.image_width else args.image_length) // vae_scale

    print(f"[INFO] Latent space: {latent_channels} x {latent_h} x {latent_w}")
    print(f"[INFO] Mode: {args.mode}, Scheduler: {args.scheduler}")

    # Reference model for CLIP Score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark with parameterized latent dimensions
    wm_kwargs = dict(
        latent_channels=latent_channels,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number, **wm_kwargs)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number, **wm_kwargs)

    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # acc
    acc = []
    # CLIP Scores
    clip_scores = []

    # For img2img mode, we need a source image pipeline
    # We generate a normal image first, then use it as source for img2img
    img_height = args.image_length
    img_width = args.image_width if args.image_width else args.image_length

    # Per-image log file (JSONL format) for dataset sensitivity analysis
    per_image_log_file = None
    if args.per_image_log:
        log_path = os.path.join(args.output_path, 'per_image_log.jsonl')
        per_image_log_file = open(log_path, 'a')
        print(f"[INFO] Per-image log: {log_path}")

    # test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        # generate with watermark
        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()

        if args.mode == 'txt2img':
            # Standard text-to-image generation
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=img_height,
                width=img_width,
                latents=init_latents_w,
            )
            image_w = outputs.images[0]

        elif args.mode == 'img2img':
            # Image-to-image mode: test watermark survival with partial noise
            # First generate a clean source image
            set_random_seed(seed + 10000)
            source_latents = torch.randn_like(init_latents_w)
            source_outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=img_height,
                width=img_width,
                latents=source_latents,
            )
            source_image = source_outputs.images[0]

            # Encode source image to latent
            source_tensor = transform_img(source_image, target_size=img_height,
                                         target_width=img_width).unsqueeze(0).to(torch.float16).to(device)
            source_latent_z0 = pipe.get_image_latents(source_tensor, sample=False)

            # Mix source latent with watermarked noise based on strength
            # At strength=1.0, fully watermarked (same as txt2img)
            # At strength=0.0, fully source image (no watermark)
            strength = args.strength
            num_inference_steps = args.num_inference_steps
            # Determine the timestep to start denoising
            start_step = int(num_inference_steps * (1 - strength))
            pipe.scheduler.set_timesteps(num_inference_steps)
            timesteps = pipe.scheduler.timesteps

            if start_step < len(timesteps):
                t_start = timesteps[start_step]
                # Add noise to source at t_start using watermarked noise
                alpha_prod_t = pipe.scheduler.alphas_cumprod[t_start]
                noisy_latent = (alpha_prod_t ** 0.5) * source_latent_z0 + \
                               ((1 - alpha_prod_t) ** 0.5) * init_latents_w
            else:
                noisy_latent = init_latents_w

            # Generate from the noisy latent
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=num_inference_steps,
                height=img_height,
                width=img_width,
                latents=noisy_latent,
            )
            image_w = outputs.images[0]

        elif args.mode == 'inpainting':
            # Inpainting mode: only part of the image is regenerated
            # Generate full watermarked image first
            outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=img_height,
                width=img_width,
                latents=init_latents_w,
            )
            watermarked_image = outputs.images[0]

            # Generate a different source image (simulating original content)
            set_random_seed(seed + 20000)
            source_latents = torch.randn_like(init_latents_w)
            source_outputs = pipe(
                current_prompt,
                num_images_per_prompt=1,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=img_height,
                width=img_width,
                latents=source_latents,
            )
            source_image = source_outputs.images[0]

            # Create a random mask and blend
            mask_ratio = args.mask_ratio
            mask = create_random_mask(img_height, img_width, mask_ratio, seed)
            # Composite: watermarked in mask region, source in non-mask region
            wm_arr = np.array(watermarked_image)
            src_arr = np.array(source_image)
            mask_3d = np.stack([mask] * 3, axis=-1)
            composite = (wm_arr * mask_3d + src_arr * (1 - mask_3d)).astype(np.uint8)
            image_w = Image.fromarray(composite)

        elif args.mode == 'controlnet':
            # ControlNet mode: generation with additional conditioning
            # Generate with standard pipeline (ControlNet changes the denoising path)
            if controlnet is not None:
                # Create a simple conditioning image (e.g., edge map from a blank canvas)
                # In practice, users provide real conditioning images
                cond_image = Image.new('RGB', (img_width, img_height), color=(128, 128, 128))
                from diffusers import StableDiffusionControlNetPipeline
                ctrl_pipe = StableDiffusionControlNetPipeline(
                    vae=pipe.vae, text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer, unet=pipe.unet,
                    controlnet=controlnet, scheduler=pipe.scheduler,
                    safety_checker=None, feature_extractor=pipe.feature_extractor,
                )
                ctrl_pipe = ctrl_pipe.to(device)
                outputs = ctrl_pipe(
                    current_prompt,
                    image=cond_image,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=img_height,
                    width=img_width,
                    latents=init_latents_w,
                )
                image_w = outputs.images[0]
            else:
                # Fallback: standard txt2img
                outputs = pipe(
                    current_prompt,
                    num_images_per_prompt=1,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    height=img_height,
                    width=img_width,
                    latents=init_latents_w,
                )
                image_w = outputs.images[0]

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # distortion
        image_w_distortion = image_distortion(image_w, seed, args)

        # Regeneration attack (if enabled)
        if args.regen_strength is not None:
            image_w_distortion = regeneration_attack(
                image_w_distortion, pipe, device,
                strength=args.regen_strength,
                prompt=current_prompt if args.regen_use_prompt else '',
                num_inference_steps=args.num_inference_steps,
            )

        # reverse img
        image_w_distortion = transform_img(image_w_distortion, target_size=img_height,
                                          target_width=img_width).unsqueeze(0).to(text_embeddings.dtype).to(device)

        # For LoRA extraction mismatch test: unload LoRA before inversion
        if args.lora_path is not None and args.lora_extract_mismatch:
            pipe.unload_lora_weights()

        # Swap to DDIM scheduler for inversion (DDIM inversion always needs DDIM math,
        # regardless of which scheduler was used for generation)
        pipe.scheduler = inversion_scheduler

        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        # Swap back to generation scheduler for next iteration
        pipe.scheduler = generation_scheduler

        # Reload LoRA if we unloaded it
        if args.lora_path is not None and args.lora_extract_mismatch:
            pipe.load_lora_weights(args.lora_path)

        # acc metric
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        acc.append(acc_metric)

        # Per-image logging
        if per_image_log_file is not None:
            log_entry = {
                'idx': i,
                'seed': seed,
                'prompt': current_prompt,
                'prompt_len': len(current_prompt.split()) if current_prompt else 0,
                'bit_acc': round(acc_metric, 6),
                'dataset': args.dataset_path,
            }
            # Include category if available (e.g., from PartiPrompts or edge_case_prompts)
            if isinstance(dataset[i], dict) and 'category' in dataset[i]:
                log_entry['category'] = dataset[i]['category']
            elif isinstance(dataset[i], dict) and 'Category' in dataset[i]:
                log_entry['category'] = dataset[i]['Category']
            elif hasattr(dataset, 'column_names') and 'Category' in dataset.column_names:
                log_entry['category'] = dataset[i]['Category']
            per_image_log_file.write(json.dumps(log_entry) + '\n')
            per_image_log_file.flush()

        # CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_preprocess,
                                              ref_tokenizer, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)

    # Close per-image log
    if per_image_log_file is not None:
        per_image_log_file.close()
        print(f"[INFO] Per-image log saved to: {os.path.join(args.output_path, 'per_image_log.jsonl')}")

    # tpr metric
    tpr_detection, tpr_traceability = watermark.get_tpr()
    # save metrics
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)

    # Print summary
    from statistics import mean, stdev
    print(f"\n{'='*60}")
    print(f"Results Summary ({args.mode}, scheduler={args.scheduler})")
    print(f"{'='*60}")
    print(f"TPR Detection:     {tpr_detection / args.num:.4f}")
    print(f"TPR Traceability:  {tpr_traceability / args.num:.4f}")
    print(f"Mean Bit Accuracy: {mean(acc):.4f}")
    print(f"Std Bit Accuracy:  {stdev(acc) if len(acc) > 1 else 0:.4f}")
    if args.reference_model is not None:
        print(f"Mean CLIP Score:   {mean(clip_scores):.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--image_width', default=None, type=int,
                        help='Image width (default: same as image_length for square images)')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--revision', default=None, help='model revision (e.g. fp16). Default: None (main branch)')

    # Scheduler selection (failure case: stochastic samplers)
    parser.add_argument('--scheduler', default='dpm', type=str,
                        choices=list(SCHEDULER_MAP.keys()),
                        help='Scheduler to use for generation')

    # Generation mode (failure cases: img2img, inpainting, controlnet)
    parser.add_argument('--mode', default='txt2img', type=str,
                        choices=['txt2img', 'img2img', 'inpainting', 'controlnet'],
                        help='Generation mode to test')
    parser.add_argument('--strength', default=0.7, type=float,
                        help='Strength for img2img mode (0.0-1.0, lower=more source image)')
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Mask ratio for inpainting mode (fraction of image to repaint)')

    # LoRA support (failure case: fine-tuned model mismatch)
    parser.add_argument('--lora_path', default=None, type=str,
                        help='Path to LoRA weights for generation')
    parser.add_argument('--lora_extract_mismatch', action='store_true',
                        help='Use base model (without LoRA) for watermark extraction')

    # ControlNet support (failure case: conditional generation)
    parser.add_argument('--controlnet_path', default=None, type=str,
                        help='Path to ControlNet model')

    # Regeneration attack (failure case: cross-model regeneration)
    parser.add_argument('--regen_strength', default=None, type=float,
                        help='Strength for regeneration attack (e.g., 0.3-0.7)')
    parser.add_argument('--regen_use_prompt', action='store_true',
                        help='Use original prompt during regeneration attack')

    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    # New distortion types
    parser.add_argument('--webp_quality', default=None, type=int,
                        help='WebP compression quality (1-100)')
    parser.add_argument('--rotation_angle', default=None, type=float,
                        help='Rotation angle in degrees')
    parser.add_argument('--color_jitter_saturation', default=None, type=float,
                        help='Saturation jitter factor')
    # Physical attack simulations
    parser.add_argument('--physical_attack', default=None, type=str,
                        choices=['print_scan', 'screen_capture', 'perspective'],
                        help='Physical attack simulation type')
    parser.add_argument('--physical_severity', default='medium', type=str,
                        choices=['mild', 'moderate', 'heavy'],
                        help='Severity level for physical attack simulation')
    parser.add_argument('--perspective_angle_x', default=5.0, type=float,
                        help='Perspective tilt angle around X axis (degrees)')
    parser.add_argument('--perspective_angle_y', default=None, type=float,
                        help='Perspective tilt angle around Y axis (degrees)')

    # Per-image logging for dataset sensitivity analysis
    parser.add_argument('--per_image_log', action='store_true',
                        help='Write per-image results (prompt, bit_acc, seed) to a JSONL file')

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
