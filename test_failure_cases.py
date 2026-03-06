"""
Unified Failure Case Test Framework for Gaussian Shading Watermarking.

This script systematically tests Gaussian Shading under various failure scenarios
that are NOT covered in the original paper, including:

1. Few-step generation (LCM/Turbo scenario)
2. Stochastic samplers (Euler Ancestral, etc.)
3. Image-to-image generation
4. Inpainting (partial regeneration)
5. Non-square images / different resolutions
6. Regeneration attack (cross-model img2img)
7. LoRA model mismatch
8. ControlNet guided generation
9. Combined attacks

Usage:
    python test_failure_cases.py --test all --model_path stabilityai/stable-diffusion-2-1-base
    python test_failure_cases.py --test few_steps --model_path stabilityai/stable-diffusion-2-1-base
    python test_failure_cases.py --test stochastic_sampler
"""

import argparse
import os
import json
import subprocess
import sys
from datetime import datetime


def run_experiment(name, cmd_args, results_dir):
    """Run a single experiment and capture output."""
    print(f"\n{'='*70}")
    print(f"  Running: {name}")
    print(f"{'='*70}")

    cmd = [sys.executable, 'run_gaussian_shading.py'] + cmd_args
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        output = result.stdout + result.stderr

        # Save full output
        output_file = os.path.join(results_dir, f"{name}.log")
        with open(output_file, 'w') as f:
            f.write(output)

        # Parse results from output
        metrics = parse_metrics(output)
        metrics['name'] = name
        metrics['command'] = ' '.join(cmd)
        metrics['returncode'] = result.returncode

        if result.returncode != 0:
            print(f"  [WARN] Non-zero exit code: {result.returncode}")
            print(f"  Stderr: {result.stderr[:500]}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout after 3600s")
        return {'name': name, 'error': 'timeout', 'command': ' '.join(cmd)}
    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return {'name': name, 'error': str(e), 'command': ' '.join(cmd)}


def parse_metrics(output):
    """Parse metrics from experiment output."""
    metrics = {}
    for line in output.split('\n'):
        if 'TPR Detection:' in line:
            try:
                metrics['tpr_detection'] = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif 'TPR Traceability:' in line:
            try:
                metrics['tpr_traceability'] = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif 'Mean Bit Accuracy:' in line:
            try:
                metrics['mean_bit_acc'] = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif 'Std Bit Accuracy:' in line:
            try:
                metrics['std_bit_acc'] = float(line.split(':')[-1].strip())
            except ValueError:
                pass
    return metrics


def get_common_args(args):
    """Get common arguments for all experiments."""
    # Use the local cache path if it exists, otherwise original dataset_path
    local_cache = os.path.join('.cache_datasets', args.dataset_path.replace('/', '_'))
    dataset_arg = local_cache if os.path.isdir(local_cache) else args.dataset_path

    base = [
        '--num', str(args.num),
        '--model_path', args.model_path,
        '--output_path', args.output_path,
        '--dataset_path', dataset_arg,
        '--chacha',
    ]
    if args.revision:
        base += ['--revision', args.revision]
    return base


# ============================================================
# Test Definitions
# ============================================================

def test_few_steps(args):
    """Test 1: Few-step generation (LCM/Turbo-like scenario).

    Hypothesis: DDIM inversion accuracy degrades significantly with very few
    inference steps (1-8), as the ODE approximation becomes too coarse.
    The paper only tests 50 steps; modern LCM/Turbo models use 1-4 steps.
    """
    experiments = []
    base = get_common_args(args)

    # Note: DPMSolver requires at least 2 steps; use DDIM for 1-step test
    step_counts = [1, 2, 4, 8, 10, 25, 50]

    for steps in step_counts:
        # Test with matched inversion steps
        name = f"few_steps_{steps}_matched"
        cmd = base + [
            '--num_inference_steps', str(steps),
            '--num_inversion_steps', str(steps),
        ]
        # DPMSolver crashes with 1 step; use DDIM instead
        if steps == 1:
            cmd += ['--scheduler', 'ddim']
        experiments.append((name, cmd))

        # Test with fixed 50-step inversion (mismatched)
        if steps != 50:
            name = f"few_steps_{steps}_inv50"
            cmd = base + [
                '--num_inference_steps', str(steps),
                '--num_inversion_steps', '50',
            ]
            if steps == 1:
                cmd += ['--scheduler', 'ddim']
            experiments.append((name, cmd))

    return experiments


def test_stochastic_sampler(args):
    """Test 2: Stochastic samplers.

    Hypothesis: Stochastic samplers (Euler Ancestral, etc.) inject random noise
    at each denoising step, making deterministic DDIM inversion impossible.
    Bit accuracy should drop to ~0.5 (random chance).
    """
    experiments = []
    base = get_common_args(args)

    schedulers = ['euler_a', 'euler', 'lms', 'pndm', 'ddim', 'dpm']

    for sched in schedulers:
        name = f"scheduler_{sched}"
        cmd = base + ['--scheduler', sched]
        experiments.append((name, cmd))

    return experiments


def test_img2img(args):
    """Test 3: Image-to-image generation.

    Hypothesis: In img2img mode, the initial latent is a noised version of
    the source image, not pure watermarked noise. With lower strength values,
    less of the watermark survives in the initial latent, degrading extraction.
    """
    experiments = []
    base = get_common_args(args)

    strengths = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for strength in strengths:
        name = f"img2img_strength_{strength}"
        cmd = base + [
            '--mode', 'img2img',
            '--strength', str(strength),
        ]
        experiments.append((name, cmd))

    return experiments


def test_inpainting(args):
    """Test 4: Inpainting with different mask sizes.

    Hypothesis: When only a fraction of the image is regenerated (inpainting),
    the watermark capacity is proportionally reduced. Small masks (10-30%)
    should cause significant watermark extraction failure.
    """
    experiments = []
    base = get_common_args(args)

    mask_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    for ratio in mask_ratios:
        name = f"inpainting_mask_{ratio}"
        cmd = base + [
            '--mode', 'inpainting',
            '--mask_ratio', str(ratio),
        ]
        experiments.append((name, cmd))

    return experiments


def test_non_square(args):
    """Test 5: Non-square images and different resolutions.

    Hypothesis: The hardcoded 64x64 latent assumption breaks for non-square
    images. Even with parameterization, different aspect ratios may affect
    watermark diffusion pattern and voting accuracy.
    """
    experiments = []
    base = get_common_args(args)

    resolutions = [
        (512, 512),    # baseline
        (768, 512),    # landscape
        (512, 768),    # portrait
        (256, 256),    # small
        (768, 768),    # larger square
    ]

    for h, w in resolutions:
        name = f"resolution_{h}x{w}"
        cmd = base + [
            '--image_length', str(h),
            '--image_width', str(w),
        ]
        experiments.append((name, cmd))

    return experiments


def test_regeneration_attack(args):
    """Test 6: Regeneration attack (img2img re-generation).

    Hypothesis: An attacker can pass a watermarked image through img2img
    with the same or different model. This fundamentally changes the latent
    representation while preserving visual content, likely destroying the watermark.
    """
    experiments = []
    base = get_common_args(args)

    strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

    for strength in strengths:
        # Without using the original prompt
        name = f"regen_attack_noprompt_s{strength}"
        cmd = base + [
            '--regen_strength', str(strength),
        ]
        experiments.append((name, cmd))

        # With using the original prompt
        name = f"regen_attack_withprompt_s{strength}"
        cmd = base + [
            '--regen_strength', str(strength),
            '--regen_use_prompt',
        ]
        experiments.append((name, cmd))

    return experiments


def test_guidance_scale_extremes(args):
    """Test 7: Extreme guidance scales.

    Hypothesis: Very high guidance scales (>15) amplify classifier-free guidance,
    creating a larger gap between generation and inversion (which uses scale=1).
    Very low scales (<3) may reduce image quality but shouldn't affect watermarks.
    """
    experiments = []
    base = get_common_args(args)

    scales = [1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0]

    for scale in scales:
        name = f"guidance_scale_{scale}"
        cmd = base + [
            '--guidance_scale', str(scale),
        ]
        experiments.append((name, cmd))

    return experiments


def test_combined_attacks(args):
    """Test 8: Combined attacks (multiple distortions applied together).

    Hypothesis: While individual attacks may be survivable, combinations of
    attacks can compound and overwhelm the watermark's robustness.
    """
    experiments = []
    base = get_common_args(args)

    # JPEG + Resize
    name = "combined_jpeg25_resize0.5"
    cmd = base + ['--jpeg_ratio', '25', '--resize_ratio', '0.5']
    experiments.append((name, cmd))

    # JPEG + Gaussian Blur
    name = "combined_jpeg25_blur4"
    cmd = base + ['--jpeg_ratio', '25', '--gaussian_blur_r', '4']
    experiments.append((name, cmd))

    # Gaussian Noise + JPEG
    name = "combined_noise0.05_jpeg25"
    cmd = base + ['--gaussian_std', '0.05', '--jpeg_ratio', '25']
    experiments.append((name, cmd))

    # Crop + Blur + JPEG
    name = "combined_crop0.6_blur4_jpeg50"
    cmd = base + ['--random_crop_ratio', '0.6', '--gaussian_blur_r', '4', '--jpeg_ratio', '50']
    experiments.append((name, cmd))

    # Brightness + Resize + JPEG
    name = "combined_bright6_resize0.5_jpeg25"
    cmd = base + ['--brightness_factor', '6', '--resize_ratio', '0.5', '--jpeg_ratio', '25']
    experiments.append((name, cmd))

    # Heavy combined: Noise + Blur + JPEG + Resize
    name = "combined_heavy"
    cmd = base + ['--gaussian_std', '0.03', '--gaussian_blur_r', '2', '--jpeg_ratio', '50', '--resize_ratio', '0.5']
    experiments.append((name, cmd))

    return experiments


def test_new_distortions(args):
    """Test 9: New distortion types not in original paper.

    Hypothesis: Modern image formats (WebP) and geometric transforms (rotation)
    may affect watermark extraction differently than the 9 original attack types.
    """
    experiments = []
    base = get_common_args(args)

    # WebP compression at various qualities
    for quality in [10, 25, 50, 75]:
        name = f"webp_q{quality}"
        cmd = base + ['--webp_quality', str(quality)]
        experiments.append((name, cmd))

    # Rotation at various angles
    for angle in [1, 5, 10, 30, 45, 90]:
        name = f"rotation_{angle}deg"
        cmd = base + ['--rotation_angle', str(angle)]
        experiments.append((name, cmd))

    # Color saturation jitter
    for sat in [0.5, 1.0, 2.0, 5.0]:
        name = f"saturation_{sat}"
        cmd = base + ['--color_jitter_saturation', str(sat)]
        experiments.append((name, cmd))

    return experiments


def test_dataset_sensitivity(args):
    """Test 10: Dataset sensitivity - different prompt distributions.

    Hypothesis: The watermark's bit accuracy may depend on the content/style
    of the generated image. Different prompt datasets produce different image
    distributions, which may reveal content-dependent fragility in DDIM inversion.
    The paper only tests on Gustavosta/Stable-Diffusion-Prompts (curated SD prompts).
    """
    experiments = []
    base = get_common_args(args)

    # Datasets to test: (name, path, num_images_override_or_None)
    datasets = [
        ('baseline_gustavosta', 'Gustavosta/Stable-Diffusion-Prompts', None),
        ('parti_prompts', 'nateraw/parti-prompts', None),
        ('diffusiondb', 'poloclub/diffusiondb', None),
        ('drawbench', 'sayakpaul/drawbench', None),
        ('edge_cases', './edge_case_prompts.json', 50),
    ]

    for ds_name, ds_path, num_override in datasets:
        name = f"dataset_{ds_name}"
        # Build command: remove existing --dataset_path from base, add new one
        cmd_filtered = []
        skip_next = False
        for arg in base:
            if skip_next:
                skip_next = False
                continue
            if arg == '--dataset_path':
                skip_next = True
                continue
            cmd_filtered.append(arg)
        cmd = cmd_filtered + ['--dataset_path', ds_path, '--per_image_log']
        # Override num if specified (e.g., edge cases have fewer prompts)
        if num_override is not None:
            # Replace --num in the command
            cmd_num = []
            skip_num = False
            for a in cmd:
                if skip_num:
                    skip_num = False
                    continue
                if a == '--num':
                    skip_num = True
                    continue
                cmd_num.append(a)
            cmd = cmd_num + ['--num', str(num_override)]
        experiments.append((name, cmd))

    return experiments


def test_physical_attacks(args):
    """Test 11: Physical attack simulations (print-scan, screen-camera capture).

    Hypothesis: Physical-world attacks (printing then scanning, photographing a
    screen) combine multiple degradations simultaneously -- color shifts, blur,
    noise, perspective distortion, resolution loss, and compression. These
    compound effects may overwhelm the watermark's robustness even when each
    individual degradation alone is survivable.
    """
    experiments = []
    base = get_common_args(args)

    # Print-and-scan at three severity levels
    for severity in ['mild', 'moderate', 'heavy']:
        name = f"physical_print_scan_{severity}"
        cmd = base + ['--physical_attack', 'print_scan',
                      '--physical_severity', severity]
        experiments.append((name, cmd))

    # Screen-camera capture at three severity levels
    for severity in ['mild', 'moderate', 'heavy']:
        name = f"physical_screen_capture_{severity}"
        cmd = base + ['--physical_attack', 'screen_capture',
                      '--physical_severity', severity]
        experiments.append((name, cmd))

    # Perspective-only at varying angles
    for angle in [1.0, 3.0, 5.0, 10.0, 15.0]:
        name = f"physical_perspective_{angle}deg"
        cmd = base + ['--physical_attack', 'perspective',
                      '--perspective_angle_x', str(angle)]
        experiments.append((name, cmd))

    # Combined: physical attack + additional digital distortion
    name = "physical_print_scan_moderate_jpeg50"
    cmd = base + ['--physical_attack', 'print_scan',
                  '--physical_severity', 'moderate', '--jpeg_ratio', '50']
    experiments.append((name, cmd))

    name = "physical_screen_capture_moderate_resize0.5"
    cmd = base + ['--physical_attack', 'screen_capture',
                  '--physical_severity', 'moderate', '--resize_ratio', '0.5']
    experiments.append((name, cmd))

    return experiments


# ============================================================
# Test Registry
# ============================================================

TEST_REGISTRY = {
    'few_steps': test_few_steps,
    'stochastic_sampler': test_stochastic_sampler,
    'img2img': test_img2img,
    'inpainting': test_inpainting,
    'non_square': test_non_square,
    'regen_attack': test_regeneration_attack,
    'guidance_scale': test_guidance_scale_extremes,
    'combined_attacks': test_combined_attacks,
    'new_distortions': test_new_distortions,
    'dataset_sensitivity': test_dataset_sensitivity,
    'physical_attacks': test_physical_attacks,
}


def load_existing_results(results_file):
    """Load existing results from a previous run for resume support."""
    if os.path.isfile(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            # Build a dict of successful results keyed by experiment name
            successful = {}
            for r in results:
                name = r.get('name', '')
                returncode = r.get('returncode', -1)
                has_metrics = 'mean_bit_acc' in r
                if returncode == 0 and has_metrics:
                    successful[name] = r
            return successful, results
        except (json.JSONDecodeError, Exception) as e:
            print(f"[WARN] Failed to load existing results: {e}")
    return {}, []


def save_results_incremental(results_file, all_results):
    """Save results after each experiment for crash recovery."""
    os.makedirs(os.path.dirname(results_file) if os.path.dirname(results_file) else '.', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian Shading Failure Case Test Framework',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--test', default='all', type=str,
                        help='Which test to run. Options:\n' +
                             '\n'.join(f'  {k}: {v.__doc__.strip().split(chr(10))[0]}'
                                      for k, v in TEST_REGISTRY.items()) +
                             '\n  all: Run all tests')
    parser.add_argument('--num', default=100, type=int,
                        help='Number of images per experiment (default: 100)')
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--output_path', default='./failure_case_results/')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run, skipping successful experiments')
    parser.add_argument('--results_file', default='./failure_case_results/results.json',
                        help='Persistent results file for resume support')

    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts',
                        help='Dataset path (will be pre-cached before experiments)')

    args = parser.parse_args()

    # Pre-download and cache the dataset before running experiments
    # This avoids each subprocess independently downloading the dataset
    print("[INFO] Pre-caching dataset to avoid repeated downloads...")
    local_cache = os.path.join('.cache_datasets', args.dataset_path.replace('/', '_'))
    if not os.path.isdir(local_cache):
        try:
            from datasets import load_dataset
            print(f"[INFO] Downloading dataset: {args.dataset_path}")
            dataset = load_dataset(args.dataset_path)['train']
            os.makedirs(os.path.dirname(local_cache), exist_ok=True)
            dataset.save_to_disk(local_cache)
            print(f"[INFO] Dataset cached to: {local_cache}")
        except Exception as e:
            print(f"[WARN] Failed to pre-cache dataset: {e}")
            print("[WARN] Experiments will attempt to download individually")
    else:
        print(f"[INFO] Dataset already cached at: {local_cache}")

    # Create results directory (use a fixed directory for logs, not timestamped)
    results_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results for resume support
    existing_successful, existing_results = load_existing_results(args.results_file)
    if args.resume and existing_successful:
        print(f"[RESUME] Found {len(existing_successful)} successful experiments from previous run")
    elif args.resume:
        print(f"[RESUME] No previous results found at {args.results_file}, running all experiments")

    # We need a temporary output_path for the subprocess (it uses this for metric files)
    subprocess_output = os.path.join(args.output_path, 'metric_files/')
    os.makedirs(subprocess_output, exist_ok=True)
    args.output_path = subprocess_output

    # Determine which tests to run
    if args.test == 'all':
        tests_to_run = list(TEST_REGISTRY.keys())
    else:
        tests_to_run = args.test.split(',')
        for t in tests_to_run:
            if t not in TEST_REGISTRY:
                print(f"Unknown test: {t}. Available: {list(TEST_REGISTRY.keys())}")
                return

    # Collect all experiments
    all_experiments = []
    for test_name in tests_to_run:
        test_fn = TEST_REGISTRY[test_name]
        experiments = test_fn(args)
        for name, cmd in experiments:
            all_experiments.append((test_name, name, cmd))

    # Filter out already-successful experiments if resuming
    experiments_to_run = []
    skipped_count = 0
    for test_name, name, cmd in all_experiments:
        if args.resume and name in existing_successful:
            skipped_count += 1
        else:
            experiments_to_run.append((test_name, name, cmd))

    total_all = len(all_experiments)
    total_run = len(experiments_to_run)

    print(f"\n{'#'*70}")
    print(f"# Gaussian Shading Failure Case Test Framework")
    print(f"# Total experiments: {total_all}")
    if args.resume:
        print(f"# Skipped (already successful): {skipped_count}")
        print(f"# Experiments to run: {total_run}")
    print(f"# Tests: {', '.join(tests_to_run)}")
    print(f"# Images per experiment: {args.num}")
    print(f"# Results file: {args.results_file}")
    print(f"{'#'*70}")

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:\n")
        for test_name, name, cmd in experiments_to_run:
            print(f"  [{test_name}] {name}:")
            print(f"    python {' '.join(cmd)}")
        if args.resume and existing_successful:
            print(f"\n[DRY RUN] Would skip {skipped_count} already-successful experiments:")
            for name in existing_successful:
                acc = existing_successful[name].get('mean_bit_acc', -1)
                print(f"    {name}: bit_acc={acc:.4f}")
        return

    # Start with existing results (keep all previous results, we'll update/add)
    all_results = list(existing_results) if args.resume else []
    # Track names already in results to avoid duplicates
    result_names = {r.get('name', '') for r in all_results}

    # Run experiments
    for idx, (test_name, name, cmd) in enumerate(experiments_to_run):
        print(f"\n[{idx+1}/{total_run}] Category: {test_name}")
        metrics = run_experiment(name, cmd, results_dir)
        metrics['category'] = test_name
        metrics['timestamp'] = datetime.now().isoformat()

        # Update or append
        if name in result_names:
            # Replace existing failed result
            all_results = [r if r.get('name') != name else metrics for r in all_results]
        else:
            all_results.append(metrics)
            result_names.add(name)

        # Save after each experiment (incremental save for crash recovery)
        save_results_incremental(args.results_file, all_results)

        # Print quick summary
        if 'mean_bit_acc' in metrics:
            acc = metrics['mean_bit_acc']
            status = "PASS" if acc >= 0.8 else "FAIL" if acc < 0.6 else "DEGRADED"
            print(f"  >> Bit Accuracy: {acc:.4f} [{status}]")

    # Print summary report (includes ALL results: previous + new)
    print(f"\n\n{'#'*70}")
    print(f"# SUMMARY REPORT (all experiments)")
    print(f"{'#'*70}")
    print(f"\n{'Category':<25} {'Experiment':<40} {'Bit Acc':>10} {'Status':>10}")
    print(f"{'-'*85}")

    failure_cases = []
    degraded_cases = []
    error_cases = []

    for r in all_results:
        name = r.get('name', 'unknown')
        category = r.get('category', 'unknown')
        acc = r.get('mean_bit_acc', -1)
        returncode = r.get('returncode', -1)

        if returncode != 0 or acc < 0:
            status = "ERROR"
            error_cases.append(r)
        elif acc >= 0.9:
            status = "PASS"
        elif acc >= 0.8:
            status = "MINOR"
        elif acc >= 0.6:
            status = "DEGRADED"
            degraded_cases.append(r)
        else:
            status = "FAIL"
            failure_cases.append(r)

        acc_str = f"{acc:.4f}" if acc >= 0 else "N/A"
        print(f"{category:<25} {name:<40} {acc_str:>10} {status:>10}")

    total_with_results = sum(1 for r in all_results if r.get('mean_bit_acc', -1) >= 0)
    print(f"\n{'='*70}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Successful runs: {total_with_results}")
    print(f"Errors: {len(error_cases)}")
    print(f"Failures (acc < 0.6): {len(failure_cases)}")
    print(f"Degraded (0.6 <= acc < 0.8): {len(degraded_cases)}")
    print(f"Results saved to: {args.results_file}")

    if failure_cases:
        print(f"\n{'!'*70}")
        print(f"  CRITICAL FAILURE CASES FOUND:")
        for r in failure_cases:
            print(f"    - {r['name']}: bit_acc={r.get('mean_bit_acc', -1):.4f}")
        print(f"{'!'*70}")

    if degraded_cases:
        print(f"\n{'*'*70}")
        print(f"  DEGRADED CASES:")
        for r in degraded_cases:
            print(f"    - {r['name']}: bit_acc={r.get('mean_bit_acc', -1):.4f}")
        print(f"{'*'*70}")

    if error_cases:
        print(f"\n{'~'*70}")
        print(f"  EXPERIMENTS WITH ERRORS (will be retried on --resume):")
        for r in error_cases:
            print(f"    - {r['name']}: returncode={r.get('returncode', 'N/A')}")
        print(f"{'~'*70}")


if __name__ == '__main__':
    main()
