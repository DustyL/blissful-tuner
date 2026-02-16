#!/bin/bash

# ============ CONFIGURATION VARIABLES ============
# User should edit these before training

# Model paths (REQUIRED - user must set these)
DIT_MODEL="/path/to/qwen_image_bf16.safetensors"
VAE_MODEL="/path/to/qwen_image_vae.safetensors"
TEXT_ENCODER="/path/to/qwen_2.5_vl_7b.safetensors"

# Dataset and training config
DATASET_CONFIG="./dataset_config.toml"
WANDB_PROJECT="qwen-image-lora"  # WandB project name

# Training parameters
MAX_TRAIN_EPOCHS=20              # Total epochs to train
SAVE_CHECKPOINT_EVERY=4          # Save every N epochs (20/4 = 5 checkpoints)
SAMPLE_EVERY_N_EPOCHS=2          # Generate samples every 2 epochs
LEARNING_RATE=1e-4               # Learning rate for LoRA
NETWORK_DIM=16                   # LoRA rank
NETWORK_ALPHA=8                  # LoRA alpha (typically dim/2)

# Hardware/memory settings
MIXED_PRECISION="bf16"           # Options: bf16, fp16, no
USE_FP8_VL=false                 # Set to true for <16GB VRAM
BLOCKS_TO_SWAP=0                 # Set to 16-45 for extreme memory saving

# ============ AUTO-GENERATED VARIABLES ============
# Generate timestamp-based run name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="qwen-lora-${TIMESTAMP}"
OUTPUT_DIR="./outputs/${RUN_NAME}"
SAMPLE_PROMPTS="./sample_prompts.txt"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "./cache"

# ============ STEP 1: CACHE LATENTS ============
echo "Step 1/3: Caching latents with VAE..."
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_MODEL}" \
    --model_version original \
    --batch_size 4 \
    --skip_existing

# ============ STEP 2: CACHE TEXT ENCODER OUTPUTS ============
echo "Step 2/3: Caching text encoder outputs..."
FP8_VL_FLAG=""
if [ "${USE_FP8_VL}" = true ]; then
    FP8_VL_FLAG="--fp8_vl"
fi

python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder "${TEXT_ENCODER}" \
    --model_version original \
    --batch_size 4 \
    --skip_existing \
    ${FP8_VL_FLAG}

# ============ STEP 3: TRAIN LORA ============
echo "Step 3/3: Starting LoRA training..."
echo "Run name: ${RUN_NAME}"
echo "Output directory: ${OUTPUT_DIR}"

# Build optional arguments
OPTIONAL_ARGS=""
if [ "${USE_FP8_VL}" = true ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --fp8_vl"
fi
if [ "${BLOCKS_TO_SWAP}" -gt 0 ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --blocks_to_swap ${BLOCKS_TO_SWAP}"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision "${MIXED_PRECISION}" \
    src/musubi_tuner/qwen_image_train_network.py \
    --dit "${DIT_MODEL}" \
    --vae "${VAE_MODEL}" \
    --text_encoder "${TEXT_ENCODER}" \
    --dataset_config "${DATASET_CONFIG}" \
    --model_version original \
    --sdpa \
    --mixed_precision "${MIXED_PRECISION}" \
    --timestep_sampling shift \
    --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --optimizer_type adamw8bit \
    --learning_rate "${LEARNING_RATE}" \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image \
    --network_dim "${NETWORK_DIM}" \
    --network_alpha "${NETWORK_ALPHA}" \
    --max_train_epochs "${MAX_TRAIN_EPOCHS}" \
    --save_every_n_epochs "${SAVE_CHECKPOINT_EVERY}" \
    --save_last_n_epochs 5 \
    --sample_prompts "${SAMPLE_PROMPTS}" \
    --sample_every_n_epochs "${SAMPLE_EVERY_N_EPOCHS}" \
    --sample_at_first \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "qwen-lora" \
    --log_with wandb \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --wandb_run_name "${RUN_NAME}" \
    --log_tracker_name "${WANDB_PROJECT}" \
    ${OPTIONAL_ARGS}

echo "Training complete! Model saved to: ${OUTPUT_DIR}"
