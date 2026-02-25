#!/bin/bash
# ==============================================================================
# Gaussian Shading Failure Case Test Scripts
# ==============================================================================
# These scripts test failure cases NOT covered in the original paper.
# Each section corresponds to a different failure scenario.
# ==============================================================================

MODEL="stabilityai/stable-diffusion-2-1-base"
NUM=100
OUTPUT="./failure_case_results/"

echo "========================================"
echo "Gaussian Shading Failure Case Tests"
echo "========================================"

# ==============================================================================
# Test 1: Few-step generation (LCM/Turbo scenario)
# Expected: Bit accuracy degrades significantly with fewer steps
# ==============================================================================
echo ""
echo "=== Test 1: Few-Step Generation ==="

for STEPS in 1 2 4 8 10 25 50; do
    echo "  Testing ${STEPS} inference steps (matched inversion)..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --num_inference_steps $STEPS --num_inversion_steps $STEPS \
        --output_path ${OUTPUT}few_steps/ --chacha
done

# Mismatched: few inference steps but 50 inversion steps
for STEPS in 1 2 4 8; do
    echo "  Testing ${STEPS} inference steps (50 inversion steps)..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --num_inference_steps $STEPS --num_inversion_steps 50 \
        --output_path ${OUTPUT}few_steps_mismatch/ --chacha
done

# ==============================================================================
# Test 2: Stochastic samplers (fundamental limitation)
# Expected: euler_a should show ~0.5 bit accuracy (random)
# ==============================================================================
echo ""
echo "=== Test 2: Stochastic Samplers ==="

for SCHED in dpm ddim euler euler_a pndm lms; do
    echo "  Testing scheduler: ${SCHED}..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --scheduler $SCHED \
        --output_path ${OUTPUT}schedulers/ --chacha
done

# ==============================================================================
# Test 3: Image-to-image generation
# Expected: Lower strength = lower bit accuracy
# ==============================================================================
echo ""
echo "=== Test 3: Img2Img Generation ==="

for STRENGTH in 0.1 0.3 0.5 0.7 0.9 1.0; do
    echo "  Testing img2img with strength=${STRENGTH}..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --mode img2img --strength $STRENGTH \
        --output_path ${OUTPUT}img2img/ --chacha
done

# ==============================================================================
# Test 4: Inpainting with different mask sizes
# Expected: Smaller masks = worse watermark extraction
# ==============================================================================
echo ""
echo "=== Test 4: Inpainting ==="

for MASK in 0.1 0.2 0.3 0.5 0.7 0.9 1.0; do
    echo "  Testing inpainting with mask_ratio=${MASK}..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --mode inpainting --mask_ratio $MASK \
        --output_path ${OUTPUT}inpainting/ --chacha
done

# ==============================================================================
# Test 5: Non-square images
# Expected: May work after parameterization, but aspect ratio effects unknown
# ==============================================================================
echo ""
echo "=== Test 5: Non-Square Images ==="

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --image_length 512 --image_width 512 \
    --output_path ${OUTPUT}resolution/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --image_length 768 --image_width 512 \
    --output_path ${OUTPUT}resolution/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --image_length 512 --image_width 768 \
    --output_path ${OUTPUT}resolution/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --image_length 256 --image_width 256 \
    --output_path ${OUTPUT}resolution/ --chacha

# ==============================================================================
# Test 6: Regeneration attack
# Expected: Even mild regeneration (strength=0.3) may destroy watermark
# ==============================================================================
echo ""
echo "=== Test 6: Regeneration Attack ==="

for STRENGTH in 0.1 0.2 0.3 0.4 0.5 0.7; do
    echo "  Testing regeneration attack with strength=${STRENGTH}..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --regen_strength $STRENGTH \
        --output_path ${OUTPUT}regen_attack/ --chacha

    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --regen_strength $STRENGTH --regen_use_prompt \
        --output_path ${OUTPUT}regen_attack_prompt/ --chacha
done

# ==============================================================================
# Test 7: Extreme guidance scales
# Expected: Very high scales (>20) may degrade inversion accuracy
# ==============================================================================
echo ""
echo "=== Test 7: Guidance Scale Extremes ==="

for SCALE in 1.0 2.0 5.0 7.5 10.0 15.0 20.0 30.0 50.0; do
    echo "  Testing guidance_scale=${SCALE}..."
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --guidance_scale $SCALE \
        --output_path ${OUTPUT}guidance_scale/ --chacha
done

# ==============================================================================
# Test 8: Combined attacks
# Expected: Combinations should be worse than individual attacks
# ==============================================================================
echo ""
echo "=== Test 8: Combined Attacks ==="

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --jpeg_ratio 25 --resize_ratio 0.5 \
    --output_path ${OUTPUT}combined/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --jpeg_ratio 25 --gaussian_blur_r 4 \
    --output_path ${OUTPUT}combined/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --gaussian_std 0.05 --jpeg_ratio 25 \
    --output_path ${OUTPUT}combined/ --chacha

python run_gaussian_shading.py --num $NUM --model_path $MODEL \
    --gaussian_std 0.03 --gaussian_blur_r 2 --jpeg_ratio 50 --resize_ratio 0.5 \
    --output_path ${OUTPUT}combined/ --chacha

# ==============================================================================
# Test 9: New distortion types
# Expected: WebP compression and rotation may affect watermark differently
# ==============================================================================
echo ""
echo "=== Test 9: New Distortions ==="

for QUALITY in 10 25 50 75; do
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --webp_quality $QUALITY \
        --output_path ${OUTPUT}new_distortions/ --chacha
done

for ANGLE in 1 5 10 30 45 90; do
    python run_gaussian_shading.py --num $NUM --model_path $MODEL \
        --rotation_angle $ANGLE \
        --output_path ${OUTPUT}new_distortions/ --chacha
done

echo ""
echo "========================================"
echo "All failure case tests completed!"
echo "Results saved to: ${OUTPUT}"
echo "========================================"
