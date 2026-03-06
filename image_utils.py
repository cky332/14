import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import random
import math
import io


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


def _solve_perspective_coeffs(src_points, dst_points):
    """Solve for the 8 coefficients of a PIL perspective transform.

    Given 4 source-destination point pairs, solves the system:
        X = (ax + by + c) / (gx + hy + 1)
        Y = (dx + ey + f) / (gx + hy + 1)
    """
    matrix = []
    for (x, y), (X, Y) in zip(src_points, dst_points):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    b = np.array([c for pt in dst_points for c in pt], dtype=np.float64)
    coeffs = np.linalg.solve(A, b)
    return tuple(coeffs.tolist())


def simulate_perspective(img, angle_x=5.0, angle_y=5.0):
    """Simulate viewing/photographing an image from a non-perpendicular angle.

    Args:
        img: PIL Image
        angle_x: tilt around horizontal axis (degrees)
        angle_y: tilt around vertical axis (degrees)
    """
    w, h = img.size
    focal = float(w)

    ax = math.radians(angle_x)
    ay = math.radians(angle_y)

    # Original corners: TL, TR, BR, BL
    corners = [(0, 0), (w, 0), (w, h), (0, h)]

    # Project each corner through rotation
    dst = []
    for x, y in corners:
        # Center coordinates
        cx, cy = x - w / 2, y - h / 2
        # Apply Y-axis rotation
        xr = cx * math.cos(ay) / (1 - cx * math.sin(ay) / focal)
        # Apply X-axis rotation
        yr = cy * math.cos(ax) / (1 - cy * math.sin(ax) / focal)
        dst.append((xr + w / 2, yr + h / 2))

    coeffs = _solve_perspective_coeffs(dst, corners)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR,
                         fillcolor=(128, 128, 128))


def simulate_print_scan(img, severity='medium'):
    """Simulate print-then-scan pipeline.

    Applies: color depth reduction -> DPI resolution loss -> blur ->
    brightness/contrast shift -> noise -> perspective -> JPEG compression.
    """
    presets = {
        'mild':     {'depth': 7, 'blur': 0.5, 'noise': 0.01, 'jpeg': 85,
                     'bright': 0.95, 'contrast': 0.95, 'angle': 0.5, 'dpi': 0.85},
        'moderate': {'depth': 6, 'blur': 1.0, 'noise': 0.02, 'jpeg': 70,
                     'bright': 0.90, 'contrast': 0.90, 'angle': 1.0, 'dpi': 0.7},
        'heavy':    {'depth': 5, 'blur': 2.0, 'noise': 0.04, 'jpeg': 50,
                     'bright': 0.85, 'contrast': 0.80, 'angle': 2.0, 'dpi': 0.5},
    }
    p = presets[severity]

    # 1. Color depth reduction
    arr = np.array(img)
    shift = 8 - p['depth']
    arr = (arr >> shift) << shift
    img = Image.fromarray(arr)

    # 2. DPI resolution loss
    orig_size = img.size
    small = img.resize((int(orig_size[0] * p['dpi']), int(orig_size[1] * p['dpi'])),
                       Image.BILINEAR)
    img = small.resize(orig_size, Image.BILINEAR)

    # 3. Gaussian blur (ink spread)
    img = img.filter(ImageFilter.GaussianBlur(radius=p['blur']))

    # 4. Brightness and contrast shift
    img = ImageEnhance.Brightness(img).enhance(p['bright'])
    img = ImageEnhance.Contrast(img).enhance(p['contrast'])

    # 5. Gaussian noise (scanner sensor)
    arr = np.array(img).astype(np.float64)
    arr += np.random.normal(0, p['noise'] * 255, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 6. Slight perspective (paper misalignment)
    img = simulate_perspective(img, angle_x=p['angle'], angle_y=p['angle'] * 0.5)

    # 7. JPEG compression (scanner output)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=p['jpeg'])
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    return img


def simulate_screen_capture(img, severity='medium'):
    """Simulate screen-display-then-camera-capture pipeline.

    Applies: gamma mismatch -> color temperature shift -> moire (heavy) ->
    blur -> resolution loss -> noise -> perspective -> JPEG compression.
    """
    presets = {
        'mild':     {'gamma': 1.1, 'temp': (5, -3, -5), 'blur': 0.3,
                     'noise': 0.01, 'angle': 0.5, 'scale': 0.9, 'jpeg': 90,
                     'moire': False},
        'moderate': {'gamma': 1.3, 'temp': (10, -5, -10), 'blur': 0.8,
                     'noise': 0.025, 'angle': 2.0, 'scale': 0.75, 'jpeg': 75,
                     'moire': False},
        'heavy':    {'gamma': 1.6, 'temp': (20, -8, -15), 'blur': 1.5,
                     'noise': 0.05, 'angle': 5.0, 'scale': 0.5, 'jpeg': 55,
                     'moire': True},
    }
    p = presets[severity]

    # 1. Gamma correction mismatch
    arr = np.array(img).astype(np.float64) / 255.0
    arr = np.power(arr, p['gamma'])
    arr = (arr * 255).astype(np.uint8)

    # 2. Color temperature shift
    r_shift, g_shift, b_shift = p['temp']
    arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + r_shift, 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(np.int16) + g_shift, 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(arr[:, :, 2].astype(np.int16) + b_shift, 0, 255).astype(np.uint8)

    # 3. Moire pattern (heavy only)
    if p['moire']:
        h, w = arr.shape[:2]
        x = np.arange(w).reshape(1, -1)
        y = np.arange(h).reshape(-1, 1)
        freq = 0.05
        moire = (np.sin(2 * math.pi * freq * x) *
                 np.sin(2 * math.pi * freq * y) * 10)
        arr = np.clip(arr.astype(np.float64) + moire[:, :, np.newaxis], 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)

    # 4. Gaussian blur (camera defocus)
    img = img.filter(ImageFilter.GaussianBlur(radius=p['blur']))

    # 5. Resolution loss
    orig_size = img.size
    small = img.resize((int(orig_size[0] * p['scale']), int(orig_size[1] * p['scale'])),
                       Image.BILINEAR)
    img = small.resize(orig_size, Image.BILINEAR)

    # 6. Gaussian noise (camera sensor)
    arr = np.array(img).astype(np.float64)
    arr += np.random.normal(0, p['noise'] * 255, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 7. Slight perspective (camera angle)
    img = simulate_perspective(img, angle_x=p['angle'], angle_y=p['angle'] * 0.7)

    # 8. JPEG compression (camera output)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=p['jpeg'])
    buf.seek(0)
    img = Image.open(buf).convert('RGB')

    return img


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

    if hasattr(args, 'physical_attack') and args.physical_attack is not None:
        severity = getattr(args, 'physical_severity', 'medium')
        if args.physical_attack == 'print_scan':
            img = simulate_print_scan(img, severity=severity)
        elif args.physical_attack == 'screen_capture':
            img = simulate_screen_capture(img, severity=severity)
        elif args.physical_attack == 'perspective':
            angle = getattr(args, 'perspective_angle_x', 5.0)
            angle_y = getattr(args, 'perspective_angle_y', None)
            img = simulate_perspective(img, angle_x=angle,
                                       angle_y=angle_y or angle * 0.7)

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
