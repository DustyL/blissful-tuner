from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

from PIL import Image

from musubi_tuner.dataset.media_utils import glob_images, glob_videos, load_video, VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "cyan")


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, list[Image.Image]]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        caption_directory: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: Optional[int] = None,
        multiple_target: bool = False,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # 0. Fail-fast: validate image_directory exists
        if not os.path.exists(image_directory):
            raise ValueError(f"image_directory does not exist: {image_directory}")
        if not os.path.isdir(image_directory):
            raise ValueError(f"image_directory is not a directory: {image_directory}")

        # 1-2. Validate caption_extension and caption_directory
        self.caption_extension, self.caption_directory = _validate_caption_config(
            caption_extension, caption_directory, image_directory, kind="image"
        )

        # 3. Glob media files (without caption filtering - we do that separately)
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory)
        logger.info(f"found {len(self.image_paths)} images")

        # 4. Check duplicate basenames (before filtering, on all media files)
        _check_duplicate_basenames(self.image_paths, kind="image")

        # 5. Filter by caption existence
        self.image_paths = _filter_paths_by_caption(
            self.image_paths,
            self.caption_extension,
            self.caption_directory,
            self.image_directory,
            kind="image",
        )

        # check if multiple-target images exist
        self.target_paths: dict[str, list[str]] = {}  # image_path -> list of target image paths

        if self.multiple_target:
            # sort by length, longer first
            sorted_image_paths = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            all_image_paths = set(glob_images(self.image_directory))  # image1.jpg, image1_1.jpg, image1_2.jpg, ...
            multiple_target_candidates = all_image_paths - set(sorted_image_paths)  # those not in the images with captions

            if len(multiple_target_candidates) > 0:
                logger.info("checking for multiple-target images")
                for image_path in sorted_image_paths:
                    image_path_no_ext = os.path.splitext(image_path)[0]

                    # find matching multiple-target images
                    potential_paths = [p for p in multiple_target_candidates if p.startswith(image_path_no_ext + "_")]

                    if potential_paths:
                        # sort by the digits (`_0000`) suffix
                        def sort_key(path):
                            path_no_ext = os.path.splitext(path)[0]
                            digits_suffix = path_no_ext.rsplit("_", 1)[-1]
                            if not digits_suffix.isdigit():
                                raise ValueError(
                                    f"Invalid digits suffix in '{path_no_ext}'. Expected a numeric suffix after '_' "
                                    f"(e.g., '_0', '_1', '_2') for proper sorting of multiple target images."
                                )
                            return int(digits_suffix)

                        potential_paths.sort(key=sort_key)
                        self.target_paths[image_path] = potential_paths

                        # remove to avoid duplicate matching
                        multiple_target_candidates.difference_update(potential_paths)

                # check the number of targets: all multiple-target images should have the same number of targets
                num_targets = 0
                for image_path, paths in self.target_paths.items():
                    if num_targets == 0:
                        num_targets = len(paths)
                    elif num_targets != len(paths):
                        logger.error(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )
                        raise ValueError(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )

                if num_targets == 0:
                    logger.error("no multiple-target images found, but multiple_target is set to True")
                    raise ValueError("no multiple-target images found, but multiple_target is set to True")

                logger.info(f"found multiple-target images, max targets per image: {num_targets}")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}

            # sort image paths for matching control images properly: longer names first
            image_paths_sorted = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            # glob control images first
            all_control_image_paths = set(glob_images(self.control_directory))

            for image_path in image_paths_sorted:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]

                # find matching control images
                potential_paths = [
                    p
                    for p in all_control_image_paths
                    if os.path.basename(p).startswith(image_basename_no_ext + ".")
                    or os.path.basename(p).startswith(image_basename_no_ext + "_")
                ]

                # remove to avoid duplicate matching
                all_control_image_paths.difference_update(potential_paths)

                if potential_paths:
                    # sort by the digits (`_0000`) suffix, prefer the one without the suffix
                    def sort_key(path):
                        basename = os.path.basename(path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        if image_basename_no_ext == basename_no_ext:  # prefer the one without suffix
                            return 0
                        digits_suffix = basename_no_ext.rsplit("_", 1)[-1]
                        if not digits_suffix.isdigit():
                            raise ValueError(f"Invalid digits suffix in {basename_no_ext}")
                        return int(digits_suffix) + 1

                    potential_paths.sort(key=sort_key)
                    if control_count_per_image is not None and len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = (
                        potential_paths[:control_count_per_image] if control_count_per_image is not None else potential_paths
                    )
            logger.info(
                f"found {len(self.control_paths)} matching control images for {'arbitrary' if control_count_per_image is None else control_count_per_image} images"
            )

            # log the distribution of number of control images
            count_of_num_control_images = {}
            for paths in self.control_paths.values():
                count = len(paths)
                if count not in count_of_num_control_images:
                    count_of_num_control_images[count] = 0
                count_of_num_control_images[count] += 1
            for count, num_images in count_of_num_control_images.items():
                logger.info(f"  {num_images} images have {count} control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        image_path = self.image_paths[idx]
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            image_paths += self.target_paths.get(image_path, [])

        images = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        caption_path = os.path.join(self.caption_directory, basename_no_ext + self.caption_extension)
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_image_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_image_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher

class ImageJsonlDatasource(ImageDatasource):
    def __init__(self, image_jsonl_file: str, control_count_per_image: Optional[int] = None, multiple_target: bool = False):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json at line {line_num}: {line} @ {self.image_jsonl_file}")
                    raise

                # Validate required image_path key
                image_path = data.get("image_path", data.get("image_path_0"))
                if not image_path:
                    raise ValueError(f"Missing 'image_path' (or 'image_path_0') at line {line_num} in {self.image_jsonl_file}")

                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        # Check for duplicate basenames (cache keys are basename-based)
        image_paths = [item.get("image_path", item.get("image_path_0")) for item in self.data]
        _check_duplicate_basenames(image_paths, kind="image")

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            if self.control_count_per_image is None:
                logger.info(f"found {len(self.data)} images with arbitrary control images per image in JSONL data")
            else:
                missing_control_images = [
                    item["image_path"]
                    for item in self.data
                    if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                ]
                if missing_control_images:
                    logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                    raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                logger.info(
                    f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data"
                )

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            while True:
                next_index = len(image_paths)  # start from 1
                next_image_path = data.get("image_path_" + str(next_index), None)
                if next_image_path is None:
                    break
                if not os.path.exists(next_image_path):
                    raise ValueError(f"multiple-target image not found: {next_image_path}")

                image_paths.append(next_image_path)

        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        caption = data["caption"]

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image or 1000):  # arbitrary large number if control_count_per_image is None
                if f"control_path_{i}" not in data:
                    break
                control_path = data[f"control_path_{i}"]
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        caption = data["caption"]
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher

class VideoDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        video = load_video(
            video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return video

    def get_control_data_from_path(
        self,
        control_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        control = load_video(
            control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return control

    def set_start_and_end_frame(self, start_frame: Optional[int], end_frame: Optional[int]):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(self, source_fps: Optional[float], target_fps: Optional[float]):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
    def __init__(
        self,
        video_directory: str,
        caption_extension: Optional[str] = None,
        caption_directory: Optional[str] = None,
        control_directory: Optional[str] = None,
    ):
        super().__init__()
        self.video_directory = video_directory
        self.control_directory = control_directory
        self.current_idx = 0

        # 0. Fail-fast: validate video_directory exists
        if not os.path.exists(video_directory):
            raise ValueError(f"video_directory does not exist: {video_directory}")
        if not os.path.isdir(video_directory):
            raise ValueError(f"video_directory is not a directory: {video_directory}")

        # 1-2. Validate caption_extension and caption_directory
        self.caption_extension, self.caption_directory = _validate_caption_config(
            caption_extension, caption_directory, video_directory, kind="video"
        )

        # 3. Glob media files
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

        # 4. Check duplicate basenames (before filtering, on all media files)
        _check_duplicate_basenames(self.video_paths, kind="video")

        # 5. Filter by caption existence
        self.video_paths = _filter_paths_by_caption(
            self.video_paths,
            self.caption_extension,
            self.caption_directory,
            self.video_directory,
            kind="video",
        )

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control videos in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for video_path in self.video_paths:
                video_basename = os.path.basename(video_path)
                # construct control path from video path
                # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mp4"
                control_path = os.path.join(self.control_directory, video_basename)
                if os.path.exists(control_path):
                    self.control_paths[video_path] = control_path
                else:
                    # use the same base name for control path
                    base_name = os.path.splitext(video_basename)[0]

                    # directory with images. for example: video_path = "vid/video.mp4" -> control_path = "control/video"
                    potential_path = os.path.join(self.control_directory, base_name)  # no extension
                    if os.path.isdir(potential_path):
                        self.control_paths[video_path] = potential_path
                    else:
                        # another extension for control path
                        # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mov"
                        for ext in VIDEO_EXTENSIONS:
                            potential_path = os.path.join(self.control_directory, base_name + ext)
                            if os.path.exists(potential_path):
                                self.control_paths[video_path] = potential_path
                                break

            logger.info(f"found {len(self.control_paths)} matching control videos/images")
            # check if all videos have matching control paths, if not, raise an error
            missing_controls = len(self.video_paths) - len(self.control_paths)
            if missing_controls > 0:
                # logger.warning(f"Could not find matching control videos/images for {missing_controls} videos")
                missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
                logger.error(
                    f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}"
                )
                raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        video_path = self.video_paths[idx]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        _, caption = self.get_caption(idx)

        control = None
        if self.control_directory is not None and video_path in self.control_paths:
            control_path = self.control_paths[video_path]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        video_path = self.video_paths[idx]
        basename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        caption_path = os.path.join(self.caption_directory, basename_no_ext + self.caption_extension)
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher

class VideoJsonlDatasource(VideoDatasource):
    def __init__(self, video_jsonl_file: str):
        super().__init__()
        self.video_jsonl_file = video_jsonl_file
        self.current_idx = 0

        # load jsonl
        logger.info(f"load video jsonl from {self.video_jsonl_file}")
        self.data = []
        with open(self.video_jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json at line {line_num}: {line} @ {self.video_jsonl_file}")
                    raise

                # Validate required video_path key
                video_path = data.get("video_path")
                if not video_path:
                    raise ValueError(f"Missing 'video_path' at line {line_num} in {self.video_jsonl_file}")

                self.data.append(data)
        logger.info(f"loaded {len(self.data)} videos")

        # Check for duplicate basenames (cache keys are basename-based)
        video_paths = [item["video_path"] for item in self.data]
        _check_duplicate_basenames(video_paths, kind="video")

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path" in item for item in self.data)
        if self.has_control:
            control_count = sum(1 for item in self.data if "control_path" in item)
            if control_count < len(self.data):
                missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
                logger.error(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
                raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
            logger.info(f"found {control_count} control videos/images in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        video_path = data["video_path"]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        caption = data["caption"]

        control = None
        if "control_path" in data and data["control_path"]:
            control_path = data["control_path"]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        video_path = data["video_path"]
        caption = data["caption"]
        return video_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_video_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


# --- blissful-tuner additions (re-homed from monolith) ---

def _check_duplicate_basenames(paths: list[str], kind: str = "image") -> None:
    """
    Check for duplicate basenames which would cause cache collisions.
    Uses casefold() for comparison because macOS APFS (case-insensitive, the default)
    performs ICU-based Unicode case folding — e.g. 'straße' and 'STRASSE' resolve to
    the same file. This matches the actual filesystem collision behavior and is
    conservative for cross-platform portability (Windows NTFS is also case-insensitive).
    Raises ValueError with examples if duplicates found.
    """
    seen: dict[str, str] = {}  # casefolded basename -> first path
    duplicate_count = 0
    examples: list[tuple[str, str, str]] = []  # (basename, path1, path2) - bounded to 3

    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        key = basename.casefold()
        if key in seen:
            duplicate_count += 1
            if len(examples) < 3:
                examples.append((basename, seen[key], path))
        else:
            seen[key] = path

    if duplicate_count:
        msg = "; ".join(f"'{b}' in both '{p1}' and '{p2}'" for b, p1, p2 in examples)
        more = f" (and {duplicate_count - len(examples)} more)" if duplicate_count > len(examples) else ""
        raise ValueError(
            f"Duplicate {kind} basenames detected (case-insensitive) - this will cause cache file collisions "
            f"(latent and TE caches are named by basename only). "
            f"Examples: {msg}{more}. "
            f"Rename files to have unique basenames."
        )

def _filter_paths_by_caption(
    paths: list[str],
    caption_extension: str,
    caption_directory: str,
    media_directory: str,
    kind: str = "image",
) -> list[str]:
    """
    Filter paths to only those with existing caption files.
    Emits one warning if some items filtered.
    Raises ValueError if all items filtered (but some existed).
    """
    total_count = len(paths)

    # Case 1: No media files found - not a caption problem
    # (Datasource already logs "found 0 images" - don't duplicate)
    if total_count == 0:
        return []

    filtered = []
    missing_count = 0
    missing_preview: list[str] = []  # Only keep first 5 for concise warnings
    caption_dir_resolved = os.path.abspath(caption_directory)

    for path in paths:
        basename_no_ext = os.path.splitext(os.path.basename(path))[0]
        caption_path = os.path.join(caption_directory, basename_no_ext + caption_extension)

        if os.path.isfile(caption_path):
            filtered.append(path)
        else:
            missing_count += 1
            if len(missing_preview) < 5:
                missing_preview.append(basename_no_ext)

    filtered_count = len(filtered)

    # Case 2: Had media but zero captions matched - hard error
    if filtered_count == 0:
        example_paths = [
            os.path.join(caption_dir_resolved, os.path.splitext(os.path.basename(paths[i]))[0] + caption_extension)
            for i in range(min(3, total_count))
        ]
        raise ValueError(
            f"No {kind}s with matching caption files found. "
            f"Found {total_count} {kind}(s) in '{media_directory}' but 0 had matching captions. "
            f"caption_extension='{caption_extension}', caption_directory='{caption_dir_resolved}'. "
            f"Expected caption files like: {example_paths}"
        )

    # Case 3: Some items filtered - warn and continue
    if missing_count > 0:
        suffix = f" (and {missing_count - len(missing_preview)} more)" if missing_count > len(missing_preview) else ""
        example_expected = [os.path.join(caption_dir_resolved, b + caption_extension) for b in missing_preview[:3]]

        # Single warning with optional >50% smell folded in
        pct_missing = missing_count / total_count * 100
        smell_note = " This may indicate a misconfiguration." if pct_missing > 50 else ""

        logger.warning(
            f"Filtered {missing_count}/{total_count} {kind}(s) without matching captions.{smell_note} "
            f"caption_extension='{caption_extension}', caption_directory='{caption_dir_resolved}'. "
            f"Missing: {missing_preview}{suffix}. "
            f"Expected paths like: {example_expected}. "
            f"Hint: If you changed captions, recache TE outputs or use a fresh cache_directory."
        )

    return filtered

def _validate_caption_config(
    caption_extension: Optional[str],
    caption_directory: Optional[str],
    media_directory: str,
    kind: str = "image",
) -> tuple[str, str]:
    """
    Validate and normalize caption_extension and caption_directory.

    Returns:
        (validated_caption_extension, effective_caption_directory)

    Raises:
        ValueError: If caption_extension is None/empty or caption_directory is invalid.
    """
    jsonl_hint = f"Use {kind}_jsonl_file if you want to embed captions directly."

    # 1. caption_extension is required for directory-based datasets
    if caption_extension is None:
        raise ValueError(f"caption_extension is required for directory-based datasets. {jsonl_hint}")

    # 2. Validate caption_extension format
    stripped = caption_extension.strip()
    if stripped == "":
        raise ValueError("caption_extension cannot be empty or whitespace")
    if stripped != caption_extension:
        logger.warning(
            f"caption_extension '{caption_extension!r}' contains leading/trailing whitespace; using stripped value '{stripped}'"
        )
        caption_extension = stripped
    if not caption_extension.startswith("."):
        logger.warning(
            f"caption_extension '{caption_extension}' does not start with '.'; "
            f"this may cause unexpected behavior (e.g., 'txt' expects files like 'footxt')"
        )

    # 3. Validate caption_directory
    effective_caption_dir = media_directory
    if caption_directory is not None:
        stripped_dir = caption_directory.strip()
        if stripped_dir == "":
            raise ValueError("caption_directory cannot be empty or whitespace")
        if stripped_dir != caption_directory:
            logger.warning(f"caption_directory contains leading/trailing whitespace: {caption_directory!r} -> {stripped_dir!r}")
            caption_directory = stripped_dir
        if not os.path.exists(caption_directory):
            raise ValueError(f"caption_directory does not exist: {caption_directory}")
        if not os.path.isdir(caption_directory):
            raise ValueError(f"caption_directory is not a directory: {caption_directory}")
        effective_caption_dir = caption_directory

    return caption_extension, effective_caption_dir
