# FLUX.2

## Overview / 概要

This document describes the usage of the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) \[dev\] architecture within the Musubi Tuner framework. FLUX.2-dev is an image generation model and edit model that can take a reference image as input.

This feature is experimental.

Latent pre-caching, training, and inference options can be found in the `--help` output. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, AE, Text Encoder models.

### FLUX.2 [dev]

- **DiT, AE**: Download from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `flux2-dev.safetensors` and `ae.safetensors`. The weights in the subfolder are in Diffusers format and cannot be used.
- **Text Encoder (Mistral 3)**: Download all the split files from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository and specify the first file (e.g., `00001-of-00010.safetensors`) in the arguments.

<details>
<summary>日本語</summary>

DiT, AE, Text Encoder のモデルをダウンロードする必要があります。

- **DiT, AE**: [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`flux2-dev.safetensors` および `ae.safetensors` を使用してください。サブフォルダ内の重みはDiffusers形式なので使用できません。
- **Text Encoder (Mistral 3)**: Download all the split files from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository and specify the first file (e.g., `00001-of-00010.safetensors`) in the arguments.
</details>

### FLUX.2 [klein] 4B / base 4B

- **DiT 4B**: Download from the [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) repository. Use `flux2-klein-4b.safetensors`.
- **DiT base 4B**: Download from the [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) repository. Use `flux2-klein-base-4b.safetensors`.
- **AE**: Download from the [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `ae.safetensors`. `vae/diffusion_pytorch_model.safetensors` in the subfolder is in Diffusers format and cannot be used.
- **Qwen3 4B Text Encoder**: Download all the split files from the [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) repository and specify the first file (e.g., `00001-of-00002.safetensors`) in the arguments.

If you already have the weights for Qwen3 4B used in Z-Image, you can use them as is. Refer to the [Z-Image documentation](./zimage.md#download-the-model--モデルのダウンロード) for details.

<details>
<summary>日本語</summary>

- **DiT 4B**: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) リポジトリからダウンロードしてください。`flux2-klein-4b.safetensors` を使用してください。
- **DiT base 4B**: [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) リポジトリからダウンロードしてください。`flux2-klein-base-4b.safetensors` を使用してください。
- **AE**: [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`ae.safetensors` を使用してください。サブフォルダ内の `vae/diffusion_pytorch_model.safetensors` はDiffusers形式なので使用できません。
- **Qwen3 4B Text Encoder**: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) リポジトリから分割されたすべてのファイルをダウンロードし、最初のファイル（例：`00001-of-00002.safetensors`）を引数で指定してください。

Qwen3 4Bの重みは、すでにZ-Imageで用いているものがあればそのまま使用可能です。[Z-Imageのドキュメント](./zimage.md#download-the-model--モデルのダウンロード)を参照してください。

</details>

### FLUX.2 [klein] 9B / base 9B

- **DiT 9B**: Download from the [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) repository. Use `flux2-klein-9b.safetensors`.
- **DiT base 9B**: Download from the [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) repository. Use `flux2-klein-base-9b.safetensors`.
- **AE**: Download from the [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `ae.safetensors`. `vae/diffusion_pytorch_model.safetensors` in the subfolder is in Diffusers format and cannot be used.
- **Qwen3 8B Text Encoder**: Download all the split files from the [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) repository and specify the first file (e.g., `00001-of-00004.safetensors`) in the arguments.

<details>
<summary>日本語</summary>

- **DiT 9B**: [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) リポジトリからダウンロードしてください。`flux2-klein-9b.safetensors` を使用してください。
- **DiT base 9B**: [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) リポジトリからダウンロードしてください。`flux2-klein-base-9b.safetensors` を使用してください。
- **AE**: [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`ae.safetensors` を使用してください。サブフォルダ内の `vae/diffusion_pytorch_model.safetensors` はDiffusers形式なので使用できません。
- **Qwen3 8B Text Encoder**: [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) リポジトリから分割されたすべてのファイルをダウンロードし、最初のファイル（例：`00001-of-00004.safetensors`）を引数で指定してください。
</details>

## Specifying Model Version / モデルバージョンの指定

When specifying the model version in various scripts, use the following options:
|type|version|sampling guidance scale|num sampling steps|distilled|
|----|--------|----|----|-----|
|flux.2-dev|`--model_version flux.2-dev`|4.0|50|guidance-distilled|
|flux.2-klein-4b|`--model_version flux.2-klein-4b`|1.0|4|step-distilled (fixed)|
|flux.2-klein-base-4b|`--model_version flux.2-klein-base-4b`|4.0|50|no (teacher model)|
|flux.2-klein-9b|`--model_version flux.2-klein-9b`|1.0|4|step-distilled (fixed)|
|flux.2-klein-base-9b|`--model_version flux.2-klein-base-9b`|4.0|50|no (teacher model)|

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for FLUX.2.

```bash
python src/musubi_tuner/flux_2_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/ae_model
```

- Note that the `--vae` argument is required, not `--ae`.
- Uses `flux_2_cache_latents.py`.
- The dataset must be an image dataset.
- Use the `--model_version` option for Flux.2 Klein training.
- The `control_images` in the dataset config is used as the reference image. See [Dataset Config](./dataset_config.md#flux1-kontext-dev) for details.

**Masked Loss Training:** To enable mask-weighted loss training (e.g., face-focused LoRA), add `mask_directory` or `alpha_mask = true` to your dataset config. The caching script will bake `mask_weights` into the latent cache files. At training time, pass `--use_mask_loss` to apply the cached weights. **Note:** for FLUX.2, `mask_weights` are cached at the model latent resolution (height//16 × width//16; e.g. 64×64 for 1024×1024), so very fine mask details will be averaged out. **Important:** Always use a fresh `cache_directory` when adding or changing mask sources. See the [Masked Loss Training Guide](./MASKED_LOSS_TRAINING_GUIDE.md) for details.

<details>
<summary>日本語</summary>

latentの事前キャッシングはFLUX.2専用のスクリプトを使用します。

- `flux_2_cache_latents.py`を使用します。
- `--ae`ではなく、`--vae`引数を指定してください。
- データセットは画像データセットである必要があります。
- データセット設定の`control_images`が参照画像として使用されます。詳細は[データセット設定](./dataset_config.md#flux1-kontext-dev)を参照してください。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
```

- Uses `flux_2_cache_text_encoder_outputs.py`.
- Requires `--text_encoder` argument
- Use the `--model_version` option for Flux.2 Klein training.
- Use `--fp8_text_encoder` option to run the Text Encoder in fp8 mode for VRAM savings.
- The larger the batch size, the more VRAM is required. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `flux_2_cache_text_encoder_outputs.py`を使用します。
- テキストエンコーダーをfp8モードで実行するための`--fp8_text_encoder`オプションを使用します。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `flux_2_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/flux_2_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux2_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_flux_2 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `flux_2_train_network.py`.
- **Requires** specifying `--vae` (not `--ae`), `--text_encoder`
- **Requires** specifying `--network_module networks.lora_flux_2`.
- `--mixed_precision bf16` is recommended for FLUX.2 training.
- `--timestep_sampling flux2_shift` is recommended for FLUX.2.
- `--training_guidance_scale`: Guidance embedding scale used during training (DEV only). Default is `1.0` (guidance-neutral). If you usually sample DEV with `--embedded_cfg_scale 4.0`, consider training with `--training_guidance_scale 4.0` to better match the inference conditioning distribution.
- Note: `flux2_shift` computes its shift from the *patchified* latent spatial dims; cross-trainer "seq_len" comparisons can be ambiguous unless you confirm whether other code uses pre- or post-patchify H×W.
- Use the `--model_version` option for Flux.2 Klein training.
- **Important**: When using reference images, `batch_size` must be 1 because reference images can have different sizes and counts per sample.
- Memory saving options like `--fp8` (for DiT) and `--fp8_text_encoder` (for Text Encoder) are available. `--fp8_scaled` is recommended when using `--fp8` for DiT.
- `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

FLUX.2の学習は専用のスクリプト`flux_2_train_network.py`を使用します。

- `flux_2_train_network.py`を使用します。
- `--vae`、`--text_encoder` を指定する必要があります。
- `--network_module networks.lora_flux_2`を指定する必要があります。
- FLUX.2の学習には`--mixed_precision bf16`を推奨します。
- FLUX.2には`--timestep_sampling flux2_shift`を推奨します。
- 注意: `flux2_shift`のshiftはパッチ化後のlatentの空間次元（H×W）から計算されます。トレーナー間で`seq_len`を比較する際は、各実装がパッチ化前/後のどちらのH×Wを使っているかを確認してください。
- `--training_guidance_scale`: 学習時のガイダンス埋め込みスケール（DEVのみ）。デフォルトは`1.0`（ガイダンス中立）です。DEVを通常`--embedded_cfg_scale 4.0`で推論する場合、推論時の条件付け分布に合わせる目的で`--training_guidance_scale 4.0`での学習を検討してください。
- `--fp8`（DiT用）や`--fp8_text_encoder`（テキストエンコーダー用、Qwen3のみ）などのメモリ節約オプションが利用可能です。`--fp8_scaled`を使用することをお勧めします。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。

</details>

## Inference / 推論

Inference uses a dedicated script `flux_2_generate_image.py`.

```bash
python src/musubi_tuner/flux_2_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder path/to/text_encoder \
    --control_image_path path/to/control_image.jpg \
    --prompt "A cat" \
    --image_size 1024 1024 --infer_steps 50 \
    --fp8_scaled \
    --save_path path/to/save/dir --output_type images \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `flux_2_generate_image.py`.
- **Requires** specifying `--vae`, `--text_encoder`
- **Requires** specifying `--control_image_path` for the reference image.
- Use the `--model_version` option for Flux.2 Klein inference. 
- `--no_resize_control`: By default, the control image is resized (pixel-capped) to the recommended resolution for FLUX.2. If you specify this option, the pixel-cap resize is skipped. Note: the image is still cropped to a model-aligned (multiple-of-16) size.

    This feature is not officially supported by FLUX.2, but it is available for experimental use.

- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower. `--fp8_text_encoder` option is available to reduce memory usage of Text Encoder (Qwen3 only). `--fp8` alone is also an option for DiT but `--fp8_scaled` potentially offers better quality.
- LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--prefer_lycoris` forces the LyCORIS backend for all weight merging; `--lycoris` is a deprecated alias.
- `--infer_steps`: If omitted, uses the model default (e.g. DEV/base: 50, klein distilled: fixed 4). For distilled klein models, steps are fixed and user-specified values are overridden.
- `--embedded_cfg_scale`: Controls the guidance scale for guidance-distilled models. If omitted, uses the model default (DEV: 4.0, klein distilled: fixed 1.0). For distilled klein models, guidance is fixed and user-specified values are overridden.
- `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.
- `--prompt_wildcards`: Path to a directory of wildcard `.txt` files. Use `__keyword__` in prompts to randomly substitute from `keyword.txt`. Supports weighted selections (e.g., `red:2.0` in wildcard files). Works in single-prompt, batch (`--from_file`), and interactive modes. Note: wildcard draws use Python's `random` module and are not tied to `--seed`, so identical seeds may produce different outputs when wildcards are involved.

<details>
<summary>日本語</summary>

FLUX.2の推論は専用のスクリプト`flux_2_generate_image.py`を使用します。

- `flux_2_generate_image.py`を使用します。
- `--vae`、`--text_encoder` を指定する必要があります。
- `--control_image_path`を指定する必要があります（参照画像）。
- `--no_resize_control`: デフォルトでは、参照画像はFLUX.2の推奨解像度にリサイズ（ピクセル上限制限）されます。このオプションを指定すると、ピクセル上限リサイズはスキップされます。ただし、モデルに合わせたサイズ（16の倍数）へのクロップは引き続き行われます。

    この機能はFLUX.2では公式にサポートされていませんが、実験的に使用可能です。

- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。またText Encoderのメモリ使用量を削減するために、`--fp8_text_encoder`オプションを指定可能です（Qwen3のみ）。DiT用に`--fp8`単独のオプションも用意されていますが、`--fp8_scaled`の方が品質が良い可能性があります。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。`--prefer_lycoris`はすべてのLoRA重みマージにLyCORISバックエンドを強制します。`--lycoris`は非推奨のエイリアスです。
- `--infer_steps`: 省略した場合はモデルのデフォルト値が使われます（例: DEV/base: 50、klein蒸留: 固定 4）。蒸留されたkleinモデルではステップ数は固定で、ユーザー指定値は警告付きで上書きされます。
- `--embedded_cfg_scale`: ガイダンス蒸留モデルのガイダンススケールです。省略した場合はモデルのデフォルト値が使われます（DEV: 4.0、klein蒸留: 固定 1.0）。蒸留されたkleinモデルではガイダンスは固定で、ユーザー指定値は警告付きで上書きされます。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。
- `--prompt_wildcards`: ワイルドカード`.txt`ファイルのディレクトリパスです。プロンプト内で`__keyword__`を使うと、`keyword.txt`からランダムに置換されます。重み付き選択（例: ファイル内の`red:2.0`）に対応しています。単一プロンプト、バッチ（`--from_file`）、インタラクティブモードで動作します。注意: ワイルドカードの選択はPythonの`random`モジュールを使用し、`--seed`とは連動しません。

</details>

## Prompting Tips

### DEV (Mistral3)

FLUX.2-DEV supports structured JSON prompting for precise control. It also benefits from camera and lighting specifications:

```json
{
  "subject": "A 35-year-old woman",
  "setting": "a sunlit Parisian café terrace",
  "action": "reading a vintage book",
  "style": "cinematic photography, shallow depth of field",
  "lighting": "golden hour, warm sunlight through leaves",
  "colors": "#F5DEB3 warm wheat tones, #8B4513 rich brown accents",
  "camera": "85mm lens, f/1.8, eye-level shot"
}
```

**Tips:**
- Use hex colors (`#F5DEB3`) for precise color specification
- Include typography details when text is needed — BFL notes that objects containing text in reality (signs, labels, screens) should have explicit quoted text, otherwise the model generates gibberish
- Specify camera parameters for photographic styles

### Klein (Qwen3)

Klein models respond better to **narrative, prose-style prompts** rather than structured formats:

```
A weathered lighthouse keeper stands at the edge of a storm-battered cliff,
his silver beard catching the last golden rays of sunset. Salt-worn hands
grip the rusty railing as waves crash below, sending up spray that catches
the dying light like scattered diamonds.
```

**Tips:**
- More descriptive, flowing language works best
- Emphasis on emotional context and atmosphere
- Lighting descriptions are particularly impactful
- Avoid structured/JSON formats (no field labels)

### Image Editing (I2I)

When using a reference image with `--control_image_path`, write concise editing instructions (50-80 words):

- Specify what changes AND what stays the same (e.g., "keep the face and lighting")
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")

## Licensing

| Model | License | Commercial Use |
|-------|---------|----------------|
| **FLUX.2-dev** | FLUX Non-Commercial License | No |
| **FLUX.2-klein-9B** | FLUX Non-Commercial License | No |
| **FLUX.2-klein-base-9B** | FLUX Non-Commercial License | No |
| **FLUX.2-klein-4B** | **Apache 2.0** | **Yes** |
| **FLUX.2-klein-base-4B** | **Apache 2.0** | **Yes** |
| **FLUX.2 VAE (ae.safetensors)** | Apache 2.0 | Yes |

> **Important:** Klein-4B and Klein-base-4B are the only commercially-usable FLUX.2 variants. LoRAs trained on a specific base model inherit its license restrictions — if commercial use is important, train on `klein-base-4b` or `klein-4b`.
