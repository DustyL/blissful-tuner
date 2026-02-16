# Implementation Plan: Add `caption_directory` Support

## Overview

Add support for a separate `caption_directory` configuration option that allows users to specify caption files in a different location than the source images/videos. This enables reusing the same image dataset with different captioning strategies for different models (e.g., WAN 2.2 vs FLUX.2 prompts).

## Motivation

Currently, captions must be co-located with images in `image_directory`. This forces users who train the same subjects on multiple models to either:
- Duplicate images across multiple directories (wastes disk space)
- Symlink images into caption directories (tedious maintenance)
- Use distinctive extensions like `.flux2.txt` (clutters the image directory)

## Design Decisions

### Scope: Caption Directory Only
- **Include:** `caption_directory` for images and videos
- **Exclude:** Separate `text_encoder_cache_directory` (deferred - current `cache_directory` approach works)

### Backward Compatibility
- `caption_directory` defaults to `image_directory` / `video_directory` when not specified
- Existing configs continue to work unchanged

### Video Caption Filtering (Bug Fix)
- **Current behavior:** Videos are NOT filtered by caption existence (will crash at runtime)
- **New behavior:** Align with images - filter videos early if `caption_extension` is set
- This is a bug fix that improves UX regardless of `caption_directory`

---

## Files to Modify

### 1. `src/musubi_tuner/dataset/config_utils.py`

**Changes:**
- Add `caption_directory: Optional[str] = None` to `ImageDatasetParams` (line ~100)
- Add `caption_directory: Optional[str] = None` to `VideoDatasetParams` (line ~121)
- Add `"caption_directory": str` to `IMAGE_DATASET_DISTINCT_SCHEMA` (line ~178)
- Add `"caption_directory": str` to `VIDEO_DATASET_DISTINCT_SCHEMA` (line ~196)
- Add `caption_directory` to info output in `generate_dataset_group_by_blueprint()` (line ~376)

### 2. `src/musubi_tuner/dataset/image_video_dataset.py`

**Changes:**

#### A. `glob_images()` function (lines 100-124)
- Add optional `caption_directory` parameter
- Use `caption_directory` (or fall back to `directory`) when filtering by captions

#### B. `glob_videos()` function (lines 127-136)
- Add optional `caption_extension` and `caption_directory` parameters
- Implement caption filtering (same logic as `glob_images`)
- Add actionable warning when videos are filtered (missing_count/total, first 20 basenames)
- Raise hard error if filtering results in 0 videos (likely misconfiguration)
- **This fixes the video caption filtering bug** (currently crashes with FileNotFoundError later)

#### C. `ImageDirectoryDatasource.__init__()` (lines 1212-1230)
- Add `caption_directory` parameter
- Store `self.caption_directory = caption_directory or image_directory`
- Pass `caption_directory` to `glob_images()`

#### D. `ImageDirectoryDatasource.get_caption()` (lines 1393-1398)
- Build caption path using `self.caption_directory` instead of deriving from `image_path`

#### E. `VideoDirectoryDatasource.__init__()` (lines 1625-1675)
- Add `caption_directory` parameter
- Store `self.caption_directory = caption_directory or video_directory`
- Call updated `glob_videos()` with caption filtering

#### F. `VideoDirectoryDatasource.get_caption()` (lines 1702-1707)
- Build caption path using `self.caption_directory` instead of deriving from `video_path`

#### G. `ImageDataset.__init__()` (lines 1981-2068)
- Add `caption_directory` parameter
- Pass through to `ImageDirectoryDatasource`

#### H. `VideoDataset.__init__()` (lines 2394-2476)
- Add `caption_directory` parameter
- Pass through to `VideoDirectoryDatasource`

### 3. `docs/dataset_config.md`

**Changes:**
- Add `caption_directory` to specification section
- Add usage example showing separate caption directories
- Note that `caption_directory` defaults to `image_directory`/`video_directory`

### 4. `src/musubi_tuner/gui/gui.py` (optional)

**Changes:**
- Add caption_directory input field (line ~313)
- Update `generate_config()` to include `caption_directory` in TOML output

### 5. `src/musubi_tuner/gui/i18n_data.py` (optional, if GUI updated)

**Changes:**
- Add localization strings for caption_directory UI

---

## Safety Improvements (Defensive Coding)

### Caption Filtering Behavior (Images & Videos)

**Default behavior:** Warning + filter (matches image semantics but with actionable feedback)

**Location:** `glob_images()` and new `glob_videos()` filtering logic

#### When some items are filtered (missing captions):
```python
if len(missing_basenames) > 0:
    preview = missing_basenames[:20]
    suffix = f" and {len(missing_basenames) - 20} more" if len(missing_basenames) > 20 else ""
    logger.warning(
        f"Filtered {len(missing_basenames)}/{total_count} {'images' if is_image else 'videos'} "
        f"without matching captions (caption_extension='{caption_extension}', "
        f"caption_directory='{caption_dir}'). "
        f"Missing: {preview}{suffix}"
    )
```

#### When filtering results in 0 items (hard error - likely misconfiguration):
```python
if len(filtered_paths) == 0:
    raise ValueError(
        f"No {'images' if is_image else 'videos'} with matching caption files found. "
        f"Check that caption_directory='{caption_dir}' contains files with extension '{caption_extension}'. "
        f"Found {total_count} {'images' if is_image else 'videos'} but 0 had matching captions."
    )
```

#### Future enhancement (not in initial scope):
- Add optional `require_caption=true` config flag for strict mode (hard error if ANY item lacks caption)
- Default remains lenient (filter + warn)

### Basename Collision Detection
**Location:** `ImageDirectoryDatasource.__init__()` after globbing

When `caption_extension` is set, detect duplicate basenames and warn:
```python
basenames = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
duplicates = [b for b in basenames if basenames.count(b) > 1]
if duplicates:
    logger.warning(f"Duplicate basenames detected: {set(duplicates)}. Caption matching may be ambiguous.")

---

## Example TOML Config (After Implementation)

```toml
[general]
caption_extension = ".txt"
batch_size = 1

# FLUX.2 training
[[datasets]]
image_directory = "/DATASETS/OLVA/images"
caption_directory = "/DATASETS/OLVA/flux2"
cache_directory = "/DATASETS/OLVA/flux2_cache"
resolution = [1024, 1024]

# WAN 2.2 training (same images, different captions)
[[datasets]]
image_directory = "/DATASETS/OLVA/images"
caption_directory = "/DATASETS/OLVA/wan"
cache_directory = "/DATASETS/OLVA/wan_cache"
resolution = [1024, 1024]
```

---

## Implementation Order

1. **Config layer** (`config_utils.py`) - Add dataclass fields and schema validation
2. **Core functions** (`image_video_dataset.py`) - Update `glob_images`, add filtering to `glob_videos`
3. **Datasources** - Update `ImageDirectoryDatasource`, `VideoDirectoryDatasource`
4. **Dataset classes** - Thread `caption_directory` through `ImageDataset`, `VideoDataset`
5. **Safety checks** - Add empty dataset and duplicate basename warnings
6. **Documentation** - Update `docs/dataset_config.md`
7. **GUI** (optional) - Add UI field and config generation

---

## Testing Verification

### Manual Test Cases

1. **Backward compatibility:**
   - Run existing config without `caption_directory` - should work unchanged

2. **Separate caption directory (images):**
   - Config with `image_directory=/A`, `caption_directory=/B`
   - Verify images load from `/A`, captions load from `/B`

3. **Separate caption directory (videos):**
   - Config with `video_directory=/A`, `caption_directory=/B`
   - Verify videos load from `/A`, captions load from `/B`

4. **Empty dataset warning:**
   - Point `caption_directory` to empty folder
   - Verify warning is logged

5. **Video caption filtering fix:**
   - Create video without matching caption
   - Verify it's filtered out (not crash at runtime)

### Test Commands
```bash
# Test config parsing
python -m musubi_tuner.dataset.config_utils test_config.toml

# Test latent caching with new config
python wan_cache_latents.py --dataset_config test_config.toml --vae ... --dry_run

# Test text encoder caching
python wan_cache_text_encoder_outputs.py --dataset_config test_config.toml --t5 ... --dry_run
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaks existing configs | Default to current behavior when `caption_directory` not set |
| Silent wrong caption matching | Add duplicate basename warning |
| Confusing empty dataset | Add explicit warning when filtering removes all items |
| Video runtime crashes | Fix by adding caption filtering to `glob_videos` |

---

## Not In Scope (Future Enhancements)

- `text_encoder_cache_directory` - Separate TE cache location
- `require_caption` config flag - Strict mode that errors if ANY item lacks caption
- JSONL datasource caption_directory - JSONL already embeds captions
- Per-image caption overrides - Beyond current scope
