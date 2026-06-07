SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

IMAGE_POSITION_OFFSET = 65536

QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Qwen3-VL conditioning feature dim = hidden_size (4096) * number of tapped activation layers (13) = 53248.
# Leaf-module constant so both the text-encoder (producer) and cache_io (writer, width assertion) agree.
IDEOGRAM4_TE_HIDDEN_SIZE = 4096
IDEOGRAM4_TE_FEATURE_DIM = IDEOGRAM4_TE_HIDDEN_SIZE * len(QWEN3_VL_ACTIVATION_LAYERS)

# Latent-cache persistence contract (grid-native): the cache stores the already-patchified, latent_norm'd
# DiT-token grid as (128, gh, gw) under the native key `latents_{gh}x{gw}_{dtype}`, so blissful's grid-native
# reader (bucket.py) loads it unchanged. The reader never reads safetensors metadata, so these flags are
# enforced by an architecture-specific preflight (preflight_ideogram4_latent_cache), NOT by the shared reader.
# Defined in this leaf module so both cache_io.py (writer) and ideogram4_utils.py (guard/preflight) can import
# them without an import cycle.
IDEOGRAM4_LATENT_NORM_METADATA_KEY = "ideogram4_latent_norm_applied"
IDEOGRAM4_LATENT_LAYOUT_KEY = "ideogram4_latent_layout"
IDEOGRAM4_LATENT_LAYOUT_GRID_CHW = "grid_chw"
IDEOGRAM4_LATENT_SPACE_KEY = "ideogram4_latent_space"
IDEOGRAM4_LATENT_SPACE_DIT_TOKENS = "ideogram4_dit_tokens"
