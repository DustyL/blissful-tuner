from __future__ import annotations

import glob
from importlib.util import find_spec
import os
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from PIL import Image
import cv2
import av

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF"]


if find_spec("jxlpy") is not None:  # JPEG-XL on Linux
    from jxlpy import JXLImagePlugin  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if find_spec("pillow_jxl") is not None:  # JPEG-XL on Windows
    import pillow_jxl  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".MP4",
    ".WEBM",
    ".AVI",
    ".MKV",
    ".MOV",
    ".FLV",
    ".WMV",
    ".M4V",
    ".MPG",
    ".MPEG",
]  # some of them are not tested


def glob_images(directory, base="*"):
    """Glob image files matching IMAGE_EXTENSIONS in a directory.

    Note: The former ``caption_extension`` parameter was removed because its
    splitext-based filtering was broken for multi-dot extensions (e.g. '.flux2.txt').
    Caption filtering is now handled by :func:`_filter_paths_by_caption`.
    """
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates
    img_paths.sort()
    return img_paths

def glob_videos(directory, base="*"):
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        if base == "*":
            video_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            video_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    video_paths = list(set(video_paths))  # remove duplicates
    video_paths.sort()
    return video_paths


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


def load_video(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    bucket_selector: Optional[BucketSelector] = None,
    bucket_reso: Optional[tuple[int, int]] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
) -> list[np.ndarray]:
    """
    bucket_reso: if given, resize the video to the bucket resolution, (width, height)
    """
    if source_fps is None or target_fps is None:
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            for i in range(len(image_files)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)
    else:
        # drop frames to match the target fps TODO commonize this code with the above if this works
        frame_index_delta = target_fps / source_fps  # example: 16 / 30 = 0.5333
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i, frame in enumerate(container.decode(video=0)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i in range(len(image_files)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)

    return video


# --- blissful-tuner additions (re-homed from monolith) ---

def resize_mask_to_bucket(mask: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize a grayscale mask to the bucket resolution.

    For weighted masks, we want:
    - Downscaling: use area averaging (preserves area/weights in expectation)
    - Upscaling: use nearest-neighbor (avoids ringing/halos and preserves discrete levels)

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(mask, Image.Image)
    if is_pil_image:
        mask_width, mask_height = mask.size
    else:
        mask_height, mask_width = mask.shape[:2]

    if bucket_reso == (mask_width, mask_height):
        return np.array(mask) if is_pil_image else mask

    bucket_width, bucket_height = bucket_reso

    scale_width = bucket_width / mask_width
    scale_height = bucket_height / mask_height
    scale = max(scale_width, scale_height)
    scaled_width = int(mask_width * scale + 0.5)
    scaled_height = int(mask_height * scale + 0.5)

    if scale > 1:
        # Upscaling: preserve discrete levels
        mask = Image.fromarray(mask) if not is_pil_image else mask
        mask = mask.resize((scaled_width, scaled_height), Image.NEAREST)
        mask = np.array(mask)
    else:
        # Downscaling: preserve average weight
        mask = np.array(mask) if is_pil_image else mask
        mask = cv2.resize(mask, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    crop_left = (scaled_width - bucket_width) // 2
    crop_top = (scaled_height - bucket_height) // 2
    mask = mask[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return mask
