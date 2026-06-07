import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ItemInfo, save_text_encoder_output_cache_ideogram4
from musubi_tuner.ideogram4.caption_verifier import verify_caption
from musubi_tuner.ideogram4.text_encoder import (
    TEXT_ENCODER_FORMATS,
    encode_prompt_to_features,
    load_ideogram4_text_encoder,
    load_ideogram4_tokenizer,
)
from musubi_tuner.utils.model_utils import str_to_dtype
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: List[ItemInfo], device, text_cache_dtype, strict_caption):
    for item in batch:
        caption = item.caption
        verify_caption(caption, strict=strict_caption)  # H4: warn-only by default
        with torch.no_grad():
            features = encode_prompt_to_features(tokenizer, text_encoder, caption, device)  # (L, 53248) float32
        save_text_encoder_output_cache_ideogram4(item, features.to(text_cache_dtype))


def ideogram4_setup_parser(parser):
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL TE weights (HF text_encoder/model.safetensors)")
    parser.add_argument("--text_encoder_config", type=str, required=True, help="local dir containing the Qwen3-VL config.json")
    parser.add_argument("--tokenizer", type=str, required=True, help="local Qwen3-VL tokenizer directory")
    parser.add_argument("--text_encoder_format", type=str, default="hf_full", choices=TEXT_ENCODER_FORMATS)
    parser.add_argument(
        "--text_cache_dtype",
        type=str,
        default="bfloat16",
        help="dtype for stored 53248-dim features (bfloat16 default; fp8_e4m3fn halves disk, a precision tradeoff)",
    )
    parser.add_argument("--strict_caption_verifier", action="store_true", help="raise on caption issues (default: warn-only, H4)")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = ideogram4_setup_parser(parser)
    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    text_cache_dtype = str_to_dtype(args.text_cache_dtype)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_IDEOGRAM4)
    datasets = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group).datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading Ideogram 4 Qwen3-VL text encoder ({args.text_encoder_format}) from {args.text_encoder}")
    text_encoder = load_ideogram4_text_encoder(
        args.text_encoder, args.text_encoder_config, text_encoder_format=args.text_encoder_format, loading_device=device
    )
    tokenizer = load_ideogram4_tokenizer(args.tokenizer)

    def encode_for_text_encoder(batch: List[ItemInfo]):
        encode_and_save_batch(tokenizer, text_encoder, batch, device, text_cache_dtype, args.strict_caption_verifier)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args,
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
