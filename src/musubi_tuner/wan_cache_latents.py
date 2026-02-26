import argparse
from typing import Optional

import torch
import torch.nn.functional as F

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from blissful_tuner.blissful_logger import BlissfulLogger

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.configs import WAN_CONFIGS, wan_i2v_14B
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.modules.clip import CLIPModel
import musubi_tuner.cache_latents as cache_latents

logger = BlissfulLogger(__name__, "green")


black_image_latents = {}  # global variable for black image latent, used in encode_and_save_batch_one_frame. key: tuple for shape


def validate_wan_cache_latents_args(args: argparse.Namespace) -> None:
    """Validate WAN cache-latents args for task-specific incompatibilities."""
    if getattr(args, "clip", None) is None:
        return

    task = getattr(args, "task", None)
    if task is None:
        return

    config = WAN_CONFIGS[task]
    if getattr(config, "v2_2", False):
        raise ValueError(
            f"--clip is specified but task '{task}' is a WAN 2.2 model which does not use CLIP. "
            "CLIP is only used for WAN 2.1 I2V caching/training. "
            "For WAN 2.2 I2V, use --i2v without --clip."
        )


def encode_and_save_batch(
    vae: WanVAE,
    clip: Optional[CLIPModel],
    i2v: bool,
    batch: list[ItemInfo],
    one_frame: bool = False,
    allow_nonconforming_frames: bool = False,
):
    if one_frame:
        encode_and_save_batch_one_frame(vae, clip, batch)
        return

    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    # LC-01: Validate T=4k+1 frame count constraint for WAN VAE temporal compression.
    # The VAE compresses frames in groups of 4 (vae_stride_t=4), requiring T=4k+1 for clean division.
    num_frames = contents.shape[2]
    if num_frames > 1 and (num_frames - 1) % 4 != 0:
        first_item = batch[0] if batch else None
        first_item_key = getattr(first_item, "item_key", None)
        more_items = f" (+{len(batch) - 1} more in batch)" if batch and len(batch) > 1 else ""
        item_hint = f" Offending item: {first_item_key}{more_items}." if first_item_key else ""
        msg = (
            f"Video frame count {num_frames} does not satisfy the T=4k+1 constraint required by the WAN VAE "
            f"(valid counts: 5, 9, 13, ..., 77, 81, 85, ...).{item_hint}"
        )
        if allow_nonconforming_frames:
            logger.warning(f"{msg} Proceeding anyway (--allow_nonconforming_frames).")
        else:
            raise ValueError(f"{msg} Use --allow_nonconforming_frames to override this check.")

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    for item in batch:
        if item.mask_content is None:
            continue
        if item.mask_content.ndim != 2:
            raise ValueError(
                f"Mask for {item.item_key} must be a 2D grayscale array (H, W), but got shape {item.mask_content.shape}."
            )
        mask_h, mask_w = item.mask_content.shape
        if (mask_h, mask_w) != (h, w):
            raise ValueError(
                f"Mask spatial dimensions (H={mask_h}, W={mask_w}) do not match content dimensions (H={h}, W={w}) for {item.item_key}. "
                "This usually means the mask was not resized/cropped to the same bucket resolution as the image/video."
            )

    # print(f"encode batch: {contents.shape}")
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        latent = vae.encode(contents)  # list of Tensor[C, F, H, W]
    latent = torch.stack(latent, dim=0)  # B, C, F, H, W
    # Autocast usually returns `vae.dtype`, but some ops may promote to fp32 for stability.
    # Ensure cached latents use the VAE dtype to avoid unexpected cache bloat / mixed dtypes downstream.
    if latent.dtype != vae.dtype:
        latent = latent.to(dtype=vae.dtype)

    if i2v:
        # extract first frame of contents
        images = contents[:, :, 0:1, :, :]  # B, C, F, H, W, non contiguous view is fine

        if clip is not None:
            with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
                clip_context = clip.visual(images)
            clip_context = clip_context.to(torch.float16)  # convert to fp16
        else:
            clip_context = None

        # encode image latent for I2V
        B, _, lat_f, lat_h, lat_w = latent.shape

        # I2V temporal mask: first latent frame = 1 (known image), rest = 0 (to generate).
        # 4 mask channels match the WAN I2V input format: 16 noisy + 4 mask + 16 image = 36 channels.
        # (The 4 corresponds to the VAE temporal scale factor / grouping.)
        msk = torch.zeros(B, 4, lat_f, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
        msk[:, :, 0] = 1

        # Pad reference image with zeros to full video length, then VAE-encode.
        # Zero-padding produces naturally decayed latents for non-reference frames.
        padding_frames = num_frames - 1
        images_padded = torch.concat([images, images.new_zeros((B, 3, padding_frames, h, w))], dim=2)
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            y = vae.encode(images_padded)
        y = torch.stack(y, dim=0)  # B, C, lat_f, H, W
        if y.dtype != vae.dtype:
            y = y.to(dtype=vae.dtype)

        # Concatenate mask + image latent → 20-channel I2V conditioning tensor
        y = torch.concat([msk, y], dim=1)  # B, 4 + C, lat_f, H, W

    else:
        clip_context = None
        y = None

    # control videos/images
    if batch[0].control_content is not None:
        # Check if control_content is a list (for images) or ndarray (for videos)
        if isinstance(batch[0].control_content, list):
            # For images with control images: control_content is list[np.ndarray]
            # We take the first control image from each item
            control_contents = torch.stack([torch.from_numpy(item.control_content[0]) for item in batch])
        else:
            # For videos with control videos: control_content is np.ndarray
            control_contents = torch.stack([torch.from_numpy(item.control_content) for item in batch])

        if len(control_contents.shape) == 4:
            control_contents = control_contents.unsqueeze(1)
        control_contents = control_contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
        control_contents = control_contents.to(vae.device, dtype=vae.dtype)
        control_contents = control_contents / 127.5 - 1.0  # normalize to [-1, 1]
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            control_latent = vae.encode(control_contents)  # list of Tensor[C, F, H, W]
        control_latent = torch.stack(control_latent, dim=0)  # B, C, F, H, W
        if control_latent.dtype != vae.dtype:
            control_latent = control_latent.to(dtype=vae.dtype)
    else:
        control_latent = None

    # # debug: decode and save
    # with torch.no_grad():
    #     latent_to_decode = latent / vae.config.scaling_factor
    #     images = vae.decode(latent_to_decode, return_dict=False)[0]
    #     images = (images / 2 + 0.5).clamp(0, 1)
    #     images = images.cpu().float().numpy()
    #     images = (images * 255).astype(np.uint8)
    #     images = images.transpose(0, 2, 3, 4, 1)  # B, C, F, H, W -> B, F, H, W, C
    #     for b in range(images.shape[0]):
    #         for f in range(images.shape[1]):
    #             fln = os.path.splitext(os.path.basename(batch[b].item_key))[0]
    #             img = Image.fromarray(images[b, f])
    #             img.save(f"./logs/decode_{fln}_{b}_{f:03d}.jpg")

    # Process mask content - downsample to latent space dimensions
    # Latent dimensions: (C, F, H/8, W/8) for WAN VAE with stride 8
    # Note: Mixed-mask batches are handled at training time in BucketBatchManager.__getitem__
    # which pads missing masks with ones for proper batch alignment
    _, _, lat_f, lat_h, lat_w = latent.shape

    for i, item in enumerate(batch):
        l = latent[i]
        cctx = clip_context[i] if clip is not None else None
        y_i = y[i] if i2v else None
        control_latent_i = control_latent[i] if control_latent is not None else None

        # Process mask for this item if it has one
        mask_weights_i = None
        if item.mask_content is not None:
            # mask_content is (H, W) grayscale numpy array with values 0-255
            mask = torch.from_numpy(item.mask_content).unsqueeze(0).unsqueeze(0)  # 1, 1, H, W

            # Normalize mask from 0-255 to 0-1
            mask = (mask.float() / 255.0).clamp_(0.0, 1.0)
            mask = cache_latents.apply_cache_mask_transforms(
                mask,
                cache_mask_gamma=float(getattr(item, "cache_mask_gamma", 1.0) or 1.0),
                cache_mask_min_weight=float(getattr(item, "cache_mask_min_weight", 0.0) or 0.0),
            )

            # Downsample mask to latent space dimensions using area interpolation
            mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # 1, 1, lat_h, lat_w

            # Expand to match latent frame count (for images, lat_f is typically 1)
            mask = mask.unsqueeze(2).expand(-1, -1, lat_f, -1, -1)  # 1, 1, F, lat_h, lat_w

            mask_weights_i = mask.squeeze(0)  # 1, F, lat_h, lat_w

        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache_wan(item, l, cctx, y_i, control_latent_i, mask_weights=mask_weights_i)


def encode_and_save_batch_one_frame(vae: WanVAE, clip: Optional[CLIPModel], batch: list[ItemInfo]):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)
    assert clip is not None, "clip is required for one frame training"

    # contents: control_content + content
    _, _, contents, content_masks = cache_latents.preprocess_contents(batch)
    contents = contents.to(vae.device, dtype=vae.dtype)  # B, C, F, H, W
    assert contents.shape[2] >= 2, "One frame training requires at least 1 control frame and 1 target frame"

    h, w = contents.shape[3], contents.shape[4]
    for item in batch:
        if item.mask_content is None:
            continue
        if item.mask_content.ndim != 2:
            raise ValueError(
                f"Mask for {item.item_key} must be a 2D grayscale array (H, W), but got shape {item.mask_content.shape}."
            )
        mask_h, mask_w = item.mask_content.shape
        if (mask_h, mask_w) != (h, w):
            raise ValueError(
                f"Mask spatial dimensions (H={mask_h}, W={mask_w}) do not match content dimensions (H={h}, W={w}) for {item.item_key}. "
                "This usually means the mask was not resized/cropped to the same bucket resolution as the image/video."
            )

    # print(f"encode batch: {contents.shape}")
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        # VAE encode: we need to encode one frame at a time because VAE encoder has stride=4 for the time dimension except for the first frame.
        latent = []
        for bi in range(contents.shape[0]):
            c = contents[bi : bi + 1]  # B, C, F, H, W, b=1
            l = []
            for f in range(c.shape[2]):  # iterate over frames
                cf = c[:, :, f : f + 1, :, :]  # B, C, 1, H, W
                l.append(vae.encode(cf)[0].unsqueeze(0))  # list of [C, 1, H, W] to [1, C, 1, H, W]
            latent.append(torch.cat(l, dim=2))  # B, C, F, H, W
        latent = torch.cat(latent, dim=0)  # B, C, F, H, W

    if latent.dtype != vae.dtype:
        latent = latent.to(dtype=vae.dtype)
    control_latent = latent[:, :, :-1, :, :]
    target_latent = latent[:, :, -1:, :, :]

    # Create black image latent for the target frame
    global black_image_latents
    shape = (1, contents.shape[1], 1, contents.shape[3], contents.shape[4])  # B=1, C, F=1, H, W
    if shape not in black_image_latents:
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            black_image_latent = vae.encode(torch.zeros(shape, device=vae.device, dtype=vae.dtype))[0]
        black_image_latent = black_image_latent.to(device="cpu", dtype=vae.dtype)
        black_image_latents[shape] = black_image_latent  # store for future use
    black_image_latent = black_image_latents[shape]  # [C, 1, H, W]

    # Vision encoding per‑item (once): use first content (first control content) because it is the start image
    num_control_images = contents.shape[2] - 1  # number of control images
    if num_control_images > 2:
        logger.error(f"One frame training requires 1 or 2 control images, but found {num_control_images} in {batch[0].item_key}. ")
        raise ValueError(
            f"One frame training requires 1 or 2 control images, but found {num_control_images} in {batch[0].item_key}."
        )

    images = contents[:, :, 0:num_control_images, :, :]  # B, C, F, H, W
    clip_context = []
    for i in range(images.shape[0]):
        with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
            clip_context.append(clip.visual(images[i : i + 1]))
    clip_context = torch.stack(clip_context, dim=0)  # B, num_control_images, N, D
    clip_context = clip_context.to(torch.float16)  # convert to fp16

    B, C, _, lat_h, lat_w = latent.shape
    for i, item in enumerate(batch):
        latent = target_latent[i]  # C, 1, H, W
        num_frames = contents.shape[2]  # number of frames
        y = torch.zeros((4 + C, num_frames, lat_h, lat_w), dtype=vae.dtype, device=vae.device)  # conditioning
        l = torch.zeros((C, num_frames, lat_h, lat_w), dtype=vae.dtype, device=vae.device)  # training latent

        # Create latent and mask for the required number of frames
        control_latent_indices = item.fp_1f_clean_indices
        target_and_control_latent_indices = control_latent_indices + [item.fp_1f_target_index]
        f_indices = sorted(target_and_control_latent_indices)

        ci = 0
        target_j = None
        for j, index in enumerate(f_indices):
            if index == item.fp_1f_target_index:
                # print(f"Set target latent. latent shape: {latent.shape}, black_image_latent shape: {black_image_latent.shape}")
                y[4:, j : j + 1, :, :] = black_image_latent
                l[:, j : j + 1, :, :] = latent  # set target latent
                target_j = j
            else:
                # print(f"Set control latent. control_latent shape: {control_latent[i, :, ci, :, :].shape}")
                y[:4, j, :, :] = 1.0  # set mask to 1.0 for the clean latent frames
                y[4:, j, :, :] = control_latent[i, :, ci, :, :]  # set control latent
                l[:, j, :, :] = control_latent[i, :, ci, :, :]  # also set control latent to training latent
                ci += 1  # increment control latent index

        cctx = clip_context[i]

        mask_weights_i = None
        if item.mask_content is not None and target_j is not None:
            mask = (torch.from_numpy(item.mask_content).unsqueeze(0).unsqueeze(0).float() / 255.0).clamp_(0.0, 1.0)  # 1, 1, H, W
            mask = cache_latents.apply_cache_mask_transforms(
                mask,
                cache_mask_gamma=float(getattr(item, "cache_mask_gamma", 1.0) or 1.0),
                cache_mask_min_weight=float(getattr(item, "cache_mask_min_weight", 0.0) or 0.0),
            )
            mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # 1, 1, lat_h, lat_w

            mask_weights_i = torch.ones(1, num_frames, lat_h, lat_w, dtype=torch.float32)
            mask_weights_i[:, target_j, :, :] = mask[0, 0]

        logger.info(f"Saving cache for item: {item.item_key} at {item.latent_cache_path}")
        logger.info(f"  control_latent_indices: {control_latent_indices}, fp_1f_target_index: {item.fp_1f_target_index}")
        logger.info(f"  y shape: {y.shape}, mask: {y[0, :, 0, 0]}, l shape: {l.shape}, clip_context shape: {cctx.shape}")
        logger.info(f"  f_indices: {f_indices}")

        save_latent_cache_wan(item, l, cctx, y, None, f_indices=f_indices, mask_weights=mask_weights_i)


def main():
    parser = cache_latents.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    validate_wan_cache_latents_args(args)

    if args.clip is not None:
        args.i2v = True

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "vae checkpoint is required"

    vae_path = args.vae

    logger.info(f"Loading VAE model from {vae_path}")
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)

    if args.clip is not None:
        clip_dtype = wan_i2v_14B.i2v_14B["clip_dtype"]
        task = getattr(args, "task", None)
        if task is not None:
            task_config = WAN_CONFIGS[task]
            clip_dtype = getattr(task_config, "clip_dtype", clip_dtype)
        clip = CLIPModel(dtype=clip_dtype, device=device, weight_path=args.clip)
    else:
        clip = None

    # Encode images
    allow_nonconforming = getattr(args, "allow_nonconforming_frames", False)

    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, clip, args.i2v, one_batch, args.one_frame, allow_nonconforming)

    cache_latents.encode_datasets(datasets, encode, args)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(WAN_CONFIGS.keys()),
        help="WAN task key (optional). Enables task-specific validation for options like --clip.",
    )
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="I2V model, encode the first frame as control image. If clip is set, this is automatically set to True",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. Required for WAN 2.1 I2V. Not used for WAN 2.2 A14B.",
    )
    parser.add_argument(
        "--one_frame",
        action="store_true",
        help="Generate cache for one frame training (single frame, single section).",
    )
    parser.add_argument(
        "--allow_nonconforming_frames",
        action="store_true",
        help="Allow video frame counts that don't satisfy the T=4k+1 constraint (5, 9, 13, ..., 81, 85, ...). "
        "Non-conforming frame counts may produce incorrect latents or reshape errors.",
    )
    return parser


if __name__ == "__main__":
    main()
