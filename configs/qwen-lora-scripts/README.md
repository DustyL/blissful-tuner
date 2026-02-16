# Qwen-Image LoRA Training Scripts

Easy-to-use scripts for training character LoRAs with Qwen-Image.

## Quick Start

1. **Prepare your dataset**
2. **Configure paths and settings**
3. **Run training**: `bash train.sh`

That's it!

---

## Step 1: Prepare Your Dataset

### Folder Structure

Create a folder for your images and captions:

```
my_dataset/
├── images/
│   ├── char001.png
│   ├── char001.txt
│   ├── char002.jpg
│   ├── char002.txt
│   ├── char003.png
│   ├── char003.txt
│   └── ... (20-50 images recommended)
└── cache/
    └── (auto-generated during training)
```

### Caption Files

Each image needs a corresponding `.txt` file with the same name:

- **Image**: `char001.png`
- **Caption**: `char001.txt`

**Caption Format**:
```
sks person wearing a blue jacket, standing in a park
```

**Tips**:
- Use a unique trigger word (e.g., "sks", or your character name)
- Describe what's in the image: pose, clothing, background, lighting
- Be consistent with your trigger word across all captions
- Keep captions descriptive but natural

### Dataset Size

- **Minimum**: 20 images
- **Recommended**: 30-50 images
- **Variety**: Different poses, expressions, angles, lighting

---

## Step 2: Configure Training Settings

### 2.1 Download Models

You need three model files (download from Hugging Face):

1. **DiT**: `qwen_image_bf16.safetensors`
   - From: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
   - Path: `split_files/diffusion_models/qwen_image_bf16.safetensors`

2. **VAE**: `qwen_image_vae.safetensors`
   - From: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
   - Path: `split_files/vae/qwen_image_vae.safetensors`

3. **Text Encoder**: `qwen_2.5_vl_7b.safetensors`
   - From: [Comfy-Org/Qwen-Image_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
   - Path: `split_files/text_encoders/qwen_2.5_vl_7b.safetensors`

### 2.2 Edit `train.sh`

Open `train.sh` and update the configuration section at the top:

```bash
# ============ REQUIRED: SET YOUR MODEL PATHS ============
DIT_MODEL="/path/to/qwen_image_bf16.safetensors"
VAE_MODEL="/path/to/qwen_image_vae.safetensors"
TEXT_ENCODER="/path/to/qwen_2.5_vl_7b.safetensors"

# ============ TRAINING SETTINGS ============
WANDB_PROJECT="qwen-image-lora"     # Your WandB project name
MAX_TRAIN_EPOCHS=20                 # Total training epochs
SAVE_CHECKPOINT_EVERY=4             # Save every 4 epochs = 5 checkpoints
LEARNING_RATE=1e-4                  # Learning rate
```

**Key Settings**:
- `MAX_TRAIN_EPOCHS`: Total training duration (start with 20)
- `SAVE_CHECKPOINT_EVERY`: Controls checkpoint frequency
  - Formula: `MAX_TRAIN_EPOCHS / 5` = 5 total checkpoints
  - Example: 20 epochs / 4 = save at epochs 4, 8, 12, 16, 20
- `LEARNING_RATE`: 1e-4 is a good starting point for LoRA
- `NETWORK_DIM`: 16 = lightweight LoRA (~50-80MB file size)

### 2.3 Edit `dataset_config.toml`

Update the paths to point to your dataset:

```toml
[[datasets]]
image_directory = "./my_dataset/images"
cache_directory = "./cache/my_dataset"
```

### 2.4 Edit `sample_prompts.txt` (Optional)

Add prompts you want to test during training:

```txt
sks person wearing a red jacket
sks person in a garden, smiling
sks person portrait, professional lighting
```

These will be generated every 2 epochs to monitor training progress.

---

## Step 3: Run Training

### 3.1 Login to WandB

```bash
wandb login
```

Enter your API key when prompted.

### 3.2 Start Training

```bash
bash train.sh
```

The script will:
1. Cache latents (VAE encoding)
2. Cache text encoder outputs
3. Train the LoRA model

### 3.3 Monitor Training

- **WandB Dashboard**: Check `https://wandb.ai` for real-time metrics
- **Local Logs**: Logs saved to `outputs/qwen-lora-TIMESTAMP/logs/`
- **Samples**: Generated images appear in WandB during training

---

## Output Structure

After training completes:

```
qwen-lora-scripts/
├── outputs/
│   └── qwen-lora-20260119-143022/          # Timestamped run folder
│       ├── qwen-lora_epoch_004.safetensors # Checkpoint 1
│       ├── qwen-lora_epoch_008.safetensors # Checkpoint 2
│       ├── qwen-lora_epoch_012.safetensors # Checkpoint 3
│       ├── qwen-lora_epoch_016.safetensors # Checkpoint 4
│       ├── qwen-lora_epoch_020.safetensors # Checkpoint 5 (final)
│       └── logs/                            # WandB/tensorboard logs
├── cache/
│   └── my_dataset/                          # Cached latents & text encoder outputs
└── ...
```

**Your trained LoRA models** are the `.safetensors` files in the output folder.

---

## Advanced Configuration

### Memory Optimization (for <24GB VRAM)

Edit `train.sh`:

```bash
USE_FP8_VL=true          # Use FP8 for text encoder (saves ~4GB)
BLOCKS_TO_SWAP=16        # Swap transformer blocks to RAM (saves ~8-12GB)
```

### Adjusting LoRA Capacity

```bash
NETWORK_DIM=32           # Higher = more capacity, larger file
NETWORK_ALPHA=16         # Typically dim/2
```

| Rank | File Size | Use Case |
|------|-----------|----------|
| 8    | ~30MB     | Lightweight, simple characters |
| 16   | ~60MB     | Recommended for most characters |
| 32   | ~120MB    | Complex styles/characters |
| 64   | ~240MB    | Maximum detail capture |

### Different Resolutions

Edit `dataset_config.toml`:

```toml
resolution = [768, 768]    # Smaller, faster training
resolution = [1024, 1024]  # Standard (default)
resolution = [1280, 1280]  # Higher quality, slower
```

---

## Troubleshooting

### "No training items found"
- Check that images and caption files have matching names
- Verify paths in `dataset_config.toml` are correct
- Ensure caching steps completed without errors

### Out of Memory (OOM)
- Enable `USE_FP8_VL=true` in `train.sh`
- Set `BLOCKS_TO_SWAP=16` or higher
- Reduce resolution to 768x768
- Use smaller LoRA rank: `NETWORK_DIM=8`

### Training too slow/fast
- **Too slow**: Increase learning rate to `2e-4`
- **Too fast (overfitting)**: Decrease to `5e-5`, add more images

### Poor quality results
- Add more training images (aim for 30-50)
- Improve caption quality (be more descriptive)
- Train for more epochs (try 30-40)
- Use higher LoRA rank (32 or 64)

---

## Tips for Best Results

1. **Dataset Quality > Quantity**
   - Better to have 30 good images than 100 poor ones
   - Variety in poses, expressions, lighting is crucial

2. **Caption Consistency**
   - Always use your trigger word ("sks person", not just "person")
   - Be descriptive but natural
   - Mention key attributes: clothing, pose, background

3. **Monitor Training**
   - Check WandB samples every 2 epochs
   - If quality peaks early, stop training (prevent overfitting)
   - Save checkpoints to compare different epochs

4. **Experiment**
   - Try different learning rates
   - Test various LoRA ranks
   - Adjust training epochs based on results

---

## Model Information

**Architecture**: Qwen-Image (text-to-image)
**Training Method**: LoRA (Low-Rank Adaptation)
**Optimizer**: AdamW 8-bit
**Scheduler**: Constant learning rate
**Precision**: BFloat16 mixed precision

---

## Credits

Built with [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)
