import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import random


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512, target_width=None):
    """Transform image to tensor, supporting non-square dimensions."""
    if target_width is None:
        target_width = target_size
    if target_size == target_width:
        tform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ]
        )
    else:
        tform = transforms.Compose(
            [
                transforms.Resize((target_size, target_width)),
                transforms.ToTensor(),
            ]
        )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def create_random_mask(height, width, mask_ratio, seed=0):
    """Create a random rectangular mask for inpainting tests."""
    np.random.seed(seed + 100)
    mask = np.zeros((height, width), dtype=np.float32)
    # Calculate mask area
    mask_area = int(height * width * mask_ratio)
    mask_h = int(np.sqrt(mask_area * height / width))
    mask_w = int(mask_area / mask_h) if mask_h > 0 else width
    mask_h = min(mask_h, height)
    mask_w = min(mask_w, width)
    # Random position
    start_y = np.random.randint(0, max(height - mask_h, 1))
    start_x = np.random.randint(0, max(width - mask_w, 1))
    mask[start_y:start_y + mask_h, start_x:start_x + mask_w] = 1.0
    return mask


def regeneration_attack(img, pipe, device, strength=0.5, prompt='',
                        num_inference_steps=50):
    """Regeneration attack: re-generate the image through the same or different model.
    This simulates an attacker using img2img to remove watermarks while preserving content."""
    # Encode the image to latent space
    img_tensor = transform_img(img).unsqueeze(0).to(torch.float16).to(device)
    latents = pipe.get_image_latents(img_tensor, sample=False)

    # Add noise to latents based on strength
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps

    start_step = int(num_inference_steps * (1 - strength))
    if start_step < len(timesteps):
        t = timesteps[start_step]
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        noise = torch.randn_like(latents)
        noisy_latents = (alpha_prod_t ** 0.5) * latents + ((1 - alpha_prod_t) ** 0.5) * noise
    else:
        noisy_latents = latents

    # Get text embedding for the prompt
    text_embeddings = pipe.get_text_embedding(prompt)

    # Denoise to generate new image
    outputs = pipe(
        prompt,
        num_images_per_prompt=1,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
        height=img_tensor.shape[2],
        width=img_tensor.shape[3],
        latents=noisy_latents,
    )
    return outputs.images[0]


def image_distortion(img, seed, args):

    if args.jpeg_ratio is not None:
        img.save(f"tmp_{args.jpeg_ratio}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp_{args.jpeg_ratio}.jpg")

    if args.random_crop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_crop_ratio)
        new_height = int(height * args.random_crop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_image = np.zeros_like(img)
        padded_image[start_y:end_y, start_x:end_x] = img[start_y:end_y, start_x:end_x]
        img = Image.fromarray(padded_image)

    if args.random_drop_ratio is not None:
        set_random_seed(seed)
        width, height, c = np.array(img).shape
        img = np.array(img)
        new_width = int(width * args.random_drop_ratio)
        new_height = int(height * args.random_drop_ratio)
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        padded_image = np.zeros_like(img[start_y:start_y + new_height, start_x:start_x + new_width])
        img[start_y:start_y + new_height, start_x:start_x + new_width] = padded_image
        img = Image.fromarray(img)

    if args.resize_ratio is not None:
        img_shape = np.array(img).shape
        resize_size = int(img_shape[0] * args.resize_ratio)
        img = transforms.Resize(size=resize_size)(img)
        img = transforms.Resize(size=img_shape[0])(img)

    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.median_blur_k is not None:
        img = img.filter(ImageFilter.MedianFilter(args.median_blur_k))


    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.sp_prob is not None:
        c,h,w = np.array(img).shape
        prob_zero = args.sp_prob / 2
        prob_one = 1 - prob_zero
        rdn = np.random.rand(c,h,w)
        img = np.where(rdn > prob_one, np.zeros_like(img), img)
        img = np.where(rdn < prob_zero, np.ones_like(img)*255, img)
        img = Image.fromarray(img)

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    # New distortion types
    if hasattr(args, 'webp_quality') and args.webp_quality is not None:
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='WebP', quality=args.webp_quality)
        buffer.seek(0)
        img = Image.open(buffer).convert('RGB')

    if hasattr(args, 'rotation_angle') and args.rotation_angle is not None:
        img = img.rotate(args.rotation_angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))

    if hasattr(args, 'color_jitter_saturation') and args.color_jitter_saturation is not None:
        img = transforms.ColorJitter(saturation=args.color_jitter_saturation)(img)

    return img


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
