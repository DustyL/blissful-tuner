# Implementation Plan: Add `caption_directory` Support (v2)

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
- **Note:** Images will now log warnings when some are filtered (previously silent)

### Video Caption Filtering (Bug Fix)
- **Current behavior:** Videos are NOT filtered by caption existence (crashes with FileNotFoundError at runtime)
- **New behavior:** Align with images - filter videos early if `caption_extension` is set
- This is a UX improvement that changes *when* failures occur (earlier, with better messages)

---

## Critical Bug Fixes in This Implementation

### 1. Multi-dot `caption_extension` Support (HIGH PRIORITY)

**Current bug:** `os.path.splitext()` breaks for extensions like `.flux2.txt`:
```python
# BROKEN: os.path.splitext("foo.flux2.txt") → ("foo.flux2", ".txt")
# Image base is "foo", caption base becomes "foo.flux2" → no match → 0 items → error
```

**Fix:** Strip the full `caption_extension` suffix instead of using `splitext`:
```python
def get_basename_without_caption_ext(filename: str, caption_extension: str) -> str:
    """Extract basename, correctly handling multi-dot extensions like .flux2.txt"""
    name = os.path.basename(filename)
    if caption_extension and name.endswith(caption_extension):
        return name[:-len(caption_extension)]
    # Fallback for image/video files: strip any extension
    return os.path.splitext(name)[0]
```

### 2. Duplicate Basename Must Be Hard Error (HIGH PRIORITY)

**Current risk:** Warning-only allows silent cache corruption:
- TE cache filenames are `{basename}_{arch}_te.safetensors` (no size disambiguator)
- `foo.jpg` + `foo.png` will collide 100% of the time
- Latent caches may also collide when sizes match

**Fix:** Hard error on duplicate image/video basenames for directory-based datasets:
```python
basenames = [os.path.splitext(os.path.basename(p))[0] for p in paths]
seen = {}
duplicates = []
for path, base in zip(paths, basenames):
    if base in seen:
        duplicates.append((base, seen[base], path))
    else:
        seen[base] = path

if duplicates:
    examples = duplicates[:3]
    msg = "; ".join(f"'{b}' appears in both {p1} and {p2}" for b, p1, p2 in examples)
    raise ValueError(
        f"Duplicate basenames detected - this will cause cache file collisions. "
        f"Examples: {msg}. "
        f"Rename files to have unique basenames or use image_jsonl_file for explicit paths."
    )
```

### 3. Caption Filtering Implementation (Safer Approach)

**Instead of:** Glob all captions → build set → filter images (O(#captions), parsing ambiguity)

**Use:** For each image/video, compute expected caption filepath and `os.path.isfile()` it:
```python
def filter_paths_by_caption(
    paths: list[str],
    caption_extension: str,
    caption_directory: str,
    kind: str = "image"  # or "video"
) -> tuple[list[str], list[str]]:
    """
    Filter paths to only those with existing caption files.
    Returns (filtered_paths, missing_basenames).
    """
    filtered = []
    missing_basenames = []

    for path in paths:
        basename_no_ext = os.path.splitext(os.path.basename(path))[0]
        caption_path = os.path.join(caption_directory, basename_no_ext + caption_extension)

        if os.path.isfile(caption_path):
            filtered.append(path)
        else:
            missing_basenames.append(basename_no_ext)

    return filtered, missing_basenames
```

This is O(#items), avoids caption globbing, and eliminates base-parsing ambiguity.

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

#### A. Add shared helper function `_filter_paths_by_caption()` (new, near top of file)
- Implements O(n) existence-check filtering
- Returns `(filtered_paths, missing_basenames)`
- Handles warning generation with truncation
- Raises hard error on 0 matches

#### B. Add shared helper function `_check_duplicate_basenames()` (new)
- O(n) duplicate detection using dict
- Raises hard error with examples on duplicates

#### C. `glob_images()` function (lines 100-124)
- Add optional `caption_directory` parameter
- Replace inline filtering with call to `_filter_paths_by_caption()`
- Add duplicate basename check when `caption_extension` is set

#### D. `glob_videos()` function (lines 127-136)
- Add optional `caption_extension` and `caption_directory` parameters
- Add call to `_filter_paths_by_caption()` (fixes the bug)
- Add duplicate basename check when `caption_extension` is set

#### E. `ImageDirectoryDatasource.__init__()` (lines 1212-1230)
- Add `caption_directory` parameter
- Add fail-fast validation: caption_directory exists and is a directory
- Add fail-fast validation: caption_extension is non-empty when set
- Store `self.caption_directory = caption_directory or image_directory`
- Pass `caption_directory` to `glob_images()`

#### F. `ImageDirectoryDatasource.get_caption()` (lines 1393-1398)
- Build caption path using `self.caption_directory` and basename

#### G. `VideoDirectoryDatasource.__init__()` (lines 1625-1675)
- Add `caption_directory` parameter
- Add fail-fast validation (same as images)
- Store `self.caption_directory = caption_directory or video_directory`
- Call updated `glob_videos()` with caption filtering

#### H. `VideoDirectoryDatasource.get_caption()` (lines 1702-1707)
- Build caption path using `self.caption_directory` and basename

#### I. `ImageDataset.__init__()` (lines 1981-2068)
- Add `caption_directory` parameter
- Pass through to `ImageDirectoryDatasource`

#### J. `VideoDataset.__init__()` (lines 2394-2476)
- Add `caption_directory` parameter
- Pass through to `VideoDirectoryDatasource`

### 3. `docs/dataset_config.md`

**Add new subsection:**
- Field definition for `caption_directory`
- Caption matching rule (explicit about multi-dot extensions)
- Examples for WAN vs FLUX with shared images
- Pitfalls/troubleshooting (0 matches error, duplicate basenames, stale TE caches, relative paths)

### 4. `src/musubi_tuner/gui/gui.py` (optional, lower priority)

**Changes:**
- Add caption_directory input field
- Update `generate_config()` to include `caption_directory` in TOML output

### 5. `src/musubi_tuner/gui/i18n_data.py` (optional, if GUI updated)

**Changes:**
- Add localization strings for caption_directory UI

---

## Fail-Fast Validation Checks

Add these checks in datasource `__init__()` methods:

```python
# 1. caption_directory must exist and be a directory (when provided)
if caption_directory is not None:
    if not os.path.exists(caption_directory):
        raise ValueError(f"caption_directory does not exist: {caption_directory}")
    if not os.path.isdir(caption_directory):
        raise ValueError(f"caption_directory is not a directory: {caption_directory}")

# 2. caption_extension must be non-empty when set
if caption_extension is not None and caption_extension.strip() == "":
    raise ValueError("caption_extension cannot be empty or whitespace")

# 3. Duplicate basenames → hard error (in glob_images/glob_videos after globbing)
# See _check_duplicate_basenames() helper

# 4. Zero matches → hard error (in _filter_paths_by_caption)
# See helper implementation
```

---

## Warning Format (Non-Spammy, Actionable)

**One warning per dataset init, not per file:**

```python
def _filter_paths_by_caption(...) -> tuple[list[str], list[str]]:
    # ... filtering logic ...

    if missing_basenames:
        preview = sorted(missing_basenames)[:20]
        suffix = f" and {len(missing_basenames) - 20} more" if len(missing_basenames) > 20 else ""

        # Include example expected caption paths for faster debugging
        example_paths = [
            os.path.join(caption_directory, b + caption_extension)
            for b in preview[:3]
        ]

        logger.warning(
            f"Filtered {len(missing_basenames)}/{total_count} {kind}s without matching captions. "
            f"caption_extension='{caption_extension}', caption_directory='{caption_directory}'. "
            f"Missing basenames: {preview}{suffix}. "
            f"Expected caption paths like: {example_paths}. "
            f"Hint: If you changed captions, recache TE outputs or use a fresh cache_directory."
        )

    # Hard error on 0 matches
    if len(filtered) == 0:
        raise ValueError(
            f"No {kind}s with matching caption files found. "
            f"caption_extension='{caption_extension}', caption_directory='{caption_directory}'. "
            f"Found {total_count} {kind}s but 0 had matching captions. "
            f"Check that caption files exist with the correct extension."
        )

    return filtered, missing_basenames
```

**Optional: Warn if filtering drops >50% of items** (strong smell of misconfiguration):
```python
if len(missing_basenames) > total_count // 2:
    logger.warning(
        f"More than 50% of {kind}s were filtered due to missing captions. "
        f"This may indicate a misconfiguration."
    )
```

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

1. **Shared helpers** - Add `_filter_paths_by_caption()`, `_check_duplicate_basenames()`
2. **Config layer** (`config_utils.py`) - Add dataclass fields and schema validation
3. **Core functions** - Update `glob_images()`, `glob_videos()`
4. **Datasources** - Update `ImageDirectoryDatasource`, `VideoDirectoryDatasource`
5. **Dataset classes** - Thread `caption_directory` through `ImageDataset`, `VideoDataset`
6. **Documentation** - Update `docs/dataset_config.md`
7. **Tests** - Add pytest suite (see below)
8. **GUI** (optional) - Add UI field and config generation

---

## Test Plan (Pytest Suite)

Create `tests/test_dataset_caption_directory.py`:

### Test Cases

#### 1. Backward compatibility (no caption_directory)
- **Setup:** tmpdir with `a.png` + `a.txt`
- **Expect:** dataset includes `a.png`, no warnings, `get_caption()` reads `a.txt`
- **Guards:** default behavior unchanged

#### 2. Separate caption directory
- **Setup:** images in `/A` (`a.png`), captions in `/B` (`a.txt`)
- **Expect:** includes `a.png`, caption read from `/B/a.txt`
- **Guards:** new feature works

#### 3. Multi-dot caption extension (CRITICAL)
- **Setup:** `a.png`, caption file `a.flux2.txt`, `caption_extension = ".flux2.txt"`
- **Expect:** item included; caption loads correctly
- **Guards:** fixes the high-risk bug

#### 4. Warnings with truncation
- **Setup:** 30 images, only 5 captions
- **Expect:** one warning; message includes `25/30`, first 20 basenames, "and 5 more"
- **Guards:** UX + anti-spam behavior

#### 5. Zero matches hard error
- **Setup:** images exist, captions dir empty
- **Expect:** raises `ValueError` with clear message including caption dir/ext and total count
- **Guards:** prevents expensive runs on misconfig

#### 6. Duplicate basename hard error
- **Setup:** `a.png` and `a.jpg`, caption exists
- **Expect:** raises `ValueError` mentioning duplicate basename + collision risk
- **Guards:** prevents cache corruption

#### 7. Video caption filtering bug fix
- **Setup:** 2 videos `v1.mp4`, `v2.mp4`; only `v1.txt` exists (in caption dir)
- **Expect:** dataset contains only `v1.mp4`, warning about `v2`
- **Guards:** ensures no regression to "crash later"

#### 8. Mixed datasets in one config
- **Setup:** TOML with one image dataset + one video dataset
- **Expect:** both build, per-dataset warnings/errors scoped correctly
- **Guards:** blueprint plumbing + schema changes

#### 9. Invalid caption_directory path
- **Setup:** `caption_directory` points to non-existent path
- **Expect:** raises `ValueError` immediately
- **Guards:** fail-fast validation

#### 10. Empty caption_extension
- **Setup:** `caption_extension = ""`
- **Expect:** raises `ValueError` immediately
- **Guards:** fail-fast validation

---

## Documentation Updates (`docs/dataset_config.md`)

### New Subsection: `caption_directory`

```markdown
### `caption_directory` (optional)

Directory containing caption files. Defaults to `image_directory` or `video_directory`.

**Only applies to directory-based datasets** (not JSONL).

#### Caption Matching Rule

Caption filename must be: `<image_basename_without_extension><caption_extension>`

Multi-dot extensions are supported:
- Image: `portrait.png`, `caption_extension = ".flux2.txt"` → Caption: `portrait.flux2.txt`
- Image: `portrait.png`, `caption_extension = ".txt"` → Caption: `portrait.txt`

#### Example: Shared Images, Different Captions per Model

```toml
# FLUX.2 with FLUX-style prompts
[[datasets]]
image_directory = "/data/OLVA/images"
caption_directory = "/data/OLVA/flux2_captions"
cache_directory = "/data/OLVA/flux2_cache"
caption_extension = ".txt"

# WAN 2.2 with WAN-style prompts
[[datasets]]
image_directory = "/data/OLVA/images"
caption_directory = "/data/OLVA/wan_captions"
cache_directory = "/data/OLVA/wan_cache"
caption_extension = ".txt"
```

#### Troubleshooting

**"No images/videos with matching caption files found" error:**
- Check that `caption_directory` contains files with the correct `caption_extension`
- Verify caption filenames match image basenames exactly (case-sensitive on Linux)

**"Duplicate basenames detected" error:**
- Multiple images share the same base name (e.g., `photo.jpg` and `photo.png`)
- This would cause cache file collisions
- Rename files to have unique basenames, or use `image_jsonl_file` for explicit paths

**Stale TE caches after changing captions:**
- Changing caption text requires recaching text encoder outputs
- Recommended: use a separate `cache_directory` per caption strategy/model

#### Path Resolution

Relative paths are resolved relative to the current working directory (CWD), not the config file location.
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaks existing configs | Default to current behavior when `caption_directory` not set |
| Multi-dot extension bug | Use suffix stripping instead of `splitext` |
| Silent cache corruption from duplicate basenames | Hard error on duplicates |
| Confusing empty dataset | Hard error with actionable message on 0 matches |
| Video runtime crashes | Fix by adding caption filtering to `glob_videos` |
| Warning spam | One warning per dataset, not per file; truncate lists |
| Performance with large caption dirs | O(n) existence checks instead of globbing captions |

---

## Behavior Changes to Document

1. **Images now warn** when some are filtered (previously silent) - minor UX improvement
2. **Videos now filter early** instead of crashing later - breaking-ish but justified UX improvement
3. **Duplicate basenames now error** - prevents cache corruption that was previously silent

---

## Not In Scope (Future Enhancements)

- `text_encoder_cache_directory` - Separate TE cache location (requires training bucket assembly changes)
- `require_caption` config flag - Strict mode that errors if ANY item lacks caption
- JSONL datasource `caption_directory` - JSONL already embeds captions
- Recursive dataset scanning with relative subpaths
- More robust cache key scheme to eliminate basename collision constraints
- Case-insensitive caption matching (cross-platform)