# Deprecation Notices

## Flags scheduled for removal in v0.14.0

The following CLI flags are deprecated and will be removed in **v0.14.0**. They continue to work in the current release but emit warnings on use.

### `--lycoris` (all generation scripts)

**Replacement:** `--prefer_lycoris`

The `--lycoris` flag was renamed to `--prefer_lycoris` to better describe its behavior (force LyCORIS backend for weight merging). Update your scripts and TOML configs:

```diff
- python wan_generate_video.py --lycoris ...
+ python wan_generate_video.py --prefer_lycoris ...
```

**Affected scripts:** `wan_generate_video.py`, `hv_generate_video.py`, `hv_1_5_generate_video.py`, `fpack_generate_video.py`, `flux_kontext_generate_image.py`, `flux_2_generate_image.py`, `zimage_generate_image.py`, `qwen_image_generate_image.py`

### `--compile_args` (WAN and HunyuanVideo generation)

**Replacement:** `--compile_backend`, `--compile_mode`, `--compile_dynamic`, `--compile_fullgraph`

The 4-tuple `--compile_args` flag has been replaced by individual flags for clarity and configurability:

```diff
- python wan_generate_video.py --compile --compile_args inductor default False False
+ python wan_generate_video.py --compile --compile_backend inductor --compile_mode default --compile_dynamic false --compile_fullgraph false
```

**Affected scripts:** `wan_generate_video.py`, `hv_generate_video.py`

### `--fp8_te` (FLUX.2 training)

**Replacement:** `--fp8_text_encoder`

The `--fp8_te` shorthand was renamed to `--fp8_text_encoder` for consistency:

```diff
- accelerate launch flux_2_train_network.py --fp8_te ...
+ accelerate launch flux_2_train_network.py --fp8_text_encoder ...
```

**Affected scripts:** `flux_2_train_network.py`
