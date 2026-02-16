# Implementation Plan: Add `caption_directory` Support (v3)

## Overview

Add support for a separate `caption_directory` configuration option that allows users to specify caption files in a different location than the source images/videos. This enables reusing the same image dataset with different captioning strategies for different models (e.g., WAN 2.2 vs FLUX.2 prompts).

## Motivation

Currently, captions must be co-located with images in `image_directory`. This forces users who train the same subjects on multiple models to either:
- Duplicate images across multiple directories (wastes disk space)
- Symlink images into caption directories (tedious maintenance)
- Use distinctive extensions like `.flux2.txt` (clutters the image directory)

---

## Core Design Decisions

### Scope
- **Include:** `caption_directory` for images and videos
- **Exclude:** `text_encoder_cache_directory`, GUI updates, recursive scanning

### Test Framework
- **Use `unittest`** (zero new dependencies) rather than pytest
- Tests go in `tests/test_dataset_caption_directory.py`

### Backward Compatibility
- `caption_directory` defaults to `image_directory` / `video_directory` when not specified
- Existing configs continue to work unchanged
- **Behavior change:** Images/videos now emit warnings when some are filtered (previously silent)
- **Behavior change:** Videos now filter early instead of crashing later
- **Behavior change:** Duplicate basenames now error (prevents silent cache corruption)

---

## Critical: Three-Way Filtering Behavior Matrix

When `caption_extension` is set, the filtering function must distinguish three cases:

| Condition | Behavior | Rationale |
|-----------|----------|-----------|
| `total_count == 0` | Log info "found 0 {images\|videos}" and return empty list (no error) | Empty/wrong media dir is a separate problem; don't misdiagnose as caption issue |
| `total_count > 0 && filtered_count == 0` | **Hard error** with actionable message | Definitely a caption misconfiguration |
| `0 < filtered_count < total_count` | **Warn** (once, with truncated list) + return filtered list | Some items usable, user should be aware of missing ones |

```python
def _filter_paths_by_caption(
    paths: list[str],
    caption_extension: str,
    caption_directory: str,
    kind: str = "image"
) -> list[str]:
    """
    Filter paths to only those with existing caption files.
    Emits one warning if some items filtered.
    Raises ValueError if all items filtered (but some existed).
    """
    total_count = len(paths)

    # Case 1: No media files found - not a caption problem
    if total_count == 0:
        logger.info(f"Found 0 {kind}s in directory")
        return []

    filtered = []
    missing_basenames = []
    caption_dir_resolved = os.path.abspath(caption_directory)

    for path in paths:
        basename_no_ext = os.path.splitext(os.path.basename(path))[0]
        caption_path = os.path.join(caption_directory, basename_no_ext + caption_extension)

        if os.path.isfile(caption_path):
            filtered.append(path)
        else:
            missing_basenames.append(basename_no_ext)

    filtered_count = len(filtered)

    # Case 2: Had media but zero captions matched - hard error
    if filtered_count == 0:
        example_paths = [
            os.path.join(caption_dir_resolved, paths[i] and os.path.splitext(os.path.basename(paths[i]))[0] + caption_extension)
            for i in range(min(3, total_count))
        ]
        raise ValueError(
            f"No {kind}s with matching caption files found. "
            f"Found {total_count} {kind}(s) but 0 had matching captions. "
            f"caption_extension='{caption_extension}', caption_directory='{caption_dir_resolved}'. "
            f"Expected caption files like: {example_paths}"
        )

    # Case 3: Some items filtered - warn and continue
    if missing_basenames:
        preview = missing_basenames[:20]  # Keep original order, don't sort
        suffix = f" and {len(missing_basenames) - 20} more" if len(missing_basenames) > 20 else ""
        example_expected = [
            os.path.join(caption_dir_resolved, b + caption_extension)
            for b in preview[:3]
        ]

        # Single warning with optional >50% smell folded in
        pct_missing = len(missing_basenames) / total_count * 100
        smell_note = " This may indicate a misconfiguration." if pct_missing > 50 else ""

        logger.warning(
            f"Filtered {len(missing_basenames)}/{total_count} {kind}(s) without matching captions.{smell_note} "
            f"caption_extension='{caption_extension}', caption_directory='{caption_dir_resolved}'. "
            f"Missing: {preview}{suffix}. "
            f"Expected paths like: {example_expected}. "
            f"Hint: If you changed captions, recache TE outputs or use a fresh cache_directory."
        )

    return filtered
```

---

## Critical: Basename Consistency Rule

**The same "basename" derivation must be used everywhere:**
- Filtering: `os.path.splitext(os.path.basename(path))[0]`
- `get_caption()`: `os.path.splitext(os.path.basename(image_path))[0]`
- Cache keys: already use `os.path.splitext(os.path.basename(item_key))[0]`

**Multi-dot extension handling:**
- We **never parse caption filenames** - the multi-dot bug is avoided by construction
- We compute `expected_caption_path = caption_directory / (media_basename + caption_extension)`
- This works correctly for `foo.bar.png` → expects `foo.bar.txt`
- This works correctly for multi-dot caption extensions: `foo.png` + `.flux2.txt` → expects `foo.flux2.txt`

**Remove the `get_basename_without_caption_ext()` helper from the plan** - it's not needed with the isfile approach.

---

## Critical: Duplicate Basename Check

**Placement:** Run on **unfiltered** media paths (before caption filtering), because:
- If duplicates exist in the media dir, they can't be safely disambiguated regardless of captions
- Running after filtering could miss duplicates where one has a caption and one doesn't

```python
def _check_duplicate_basenames(paths: list[str], kind: str = "image") -> None:
    """
    Check for duplicate basenames which would cause cache collisions.
    Raises ValueError with examples if duplicates found.
    """
    seen: dict[str, str] = {}  # basename -> first path
    duplicates: list[tuple[str, str, str]] = []  # (basename, path1, path2)

    for path in paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename in seen:
            duplicates.append((basename, seen[basename], path))
        else:
            seen[basename] = path

    if duplicates:
        examples = duplicates[:3]
        msg = "; ".join(f"'{b}' in both '{p1}' and '{p2}'" for b, p1, p2 in examples)
        more = f" (and {len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        raise ValueError(
            f"Duplicate {kind} basenames detected - this will cause cache file collisions. "
            f"Examples: {msg}{more}. "
            f"Rename files to have unique basenames or use {'image' if kind == 'image' else 'video'}_jsonl_file for explicit paths."
        )
```

---

## Validation Order (Critical)

Validation must happen in this order to produce correct error messages:

1. **Validate `caption_directory` exists** (if provided) - before calling glob
2. **Validate `caption_extension` is non-empty** (if provided) - before calling glob
3. **Glob media files** - returns raw paths
4. **Check duplicate basenames** - on unfiltered paths
5. **Filter by caption existence** - three-way behavior

```python
# In ImageDirectoryDatasource.__init__():

# 1. Validate caption_directory
effective_caption_dir = caption_directory if caption_directory else image_directory
if caption_directory is not None:
    if not os.path.exists(caption_directory):
        raise ValueError(f"caption_directory does not exist: {caption_directory}")
    if not os.path.isdir(caption_directory):
        raise ValueError(f"caption_directory is not a directory: {caption_directory}")

# 2. Validate caption_extension
if caption_extension is not None:
    stripped = caption_extension.strip()
    if stripped == "":
        raise ValueError("caption_extension cannot be empty or whitespace")
    if stripped != caption_extension:
        logger.warning(
            f"caption_extension '{caption_extension}' contains leading/trailing whitespace; "
            f"using stripped value '{stripped}'"
        )
        caption_extension = stripped
    if not caption_extension.startswith("."):
        logger.warning(
            f"caption_extension '{caption_extension}' does not start with '.'; "
            f"this may cause unexpected behavior (e.g., 'txt' expects files like 'footxt')"
        )

# 3. Glob media files
self.image_paths = glob_images(self.image_directory)

# 4. Check duplicate basenames (before filtering)
if caption_extension is not None:
    _check_duplicate_basenames(self.image_paths, kind="image")

# 5. Filter by caption existence
if caption_extension is not None:
    self.image_paths = _filter_paths_by_caption(
        self.image_paths, caption_extension, effective_caption_dir, kind="image"
    )
```

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

#### A. Add helper functions near top of file (after imports)
- `_filter_paths_by_caption()` - O(n) existence-check filtering with three-way behavior
- `_check_duplicate_basenames()` - O(n) duplicate detection

#### B. `glob_images()` function (lines 100-124)
- **Remove** inline caption filtering logic (move to datasource)
- **Remove** `caption_extension` parameter
- Keep only the media file globbing
- Add `caption_directory` parameter (passed through but not used here - filtering happens in datasource)

Actually, simpler approach: **leave `glob_images()` unchanged** (keep its current behavior for backward compat) and do all filtering in datasource `__init__`. This avoids changing the function signature which might affect other callers.

#### C. `glob_videos()` function (lines 127-136)
- **Leave unchanged** - filtering happens in datasource

#### D. `ImageDirectoryDatasource.__init__()` (lines 1212-1230)
- Add `caption_directory` parameter
- Add validation in correct order (see above)
- Store `self.caption_directory = caption_directory or image_directory`
- Call `_check_duplicate_basenames()` after globbing
- Call `_filter_paths_by_caption()` after duplicate check

#### E. `ImageDirectoryDatasource.get_caption()` (lines 1393-1398)
- Build caption path using `self.caption_directory`:
```python
def get_caption(self, idx: int) -> tuple[str, str]:
    image_path = self.image_paths[idx]
    basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    caption_path = os.path.join(self.caption_directory, basename_no_ext + self.caption_extension)
    with open(caption_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()
    return image_path, caption
```

#### F. `VideoDirectoryDatasource.__init__()` (lines 1625-1675)
- Add `caption_directory` parameter
- Add validation (same as images)
- Store `self.caption_directory = caption_directory or video_directory`
- Call `_check_duplicate_basenames()` after globbing
- Call `_filter_paths_by_caption()` after duplicate check
- **This fixes the video caption filtering bug**

#### G. `VideoDirectoryDatasource.get_caption()` (lines 1702-1707)
- Build caption path using `self.caption_directory` (same pattern as images)

#### H. `ImageDataset.__init__()` (lines 1981-2068)
- Add `caption_directory` parameter
- Pass through to `ImageDirectoryDatasource`

#### I. `VideoDataset.__init__()` (lines 2394-2476)
- Add `caption_directory` parameter
- Pass through to `VideoDirectoryDatasource`

### 3. `docs/dataset_config.md`

**Add new subsection with:**
- Field definition for `caption_directory`
- Caption matching rule with explicit multi-dot examples
- Examples for WAN vs FLUX with shared images
- Troubleshooting section covering all error cases
- Platform-dependent case sensitivity note
- Path resolution policy (CWD, recommend absolute paths)
- Non-recursive behavior note
- Cache directory uniqueness gotcha

### 4. `tests/test_dataset_caption_directory.py` (new file)

**Use `unittest` framework** (no new dependencies)

---

## Test Plan (unittest)

### Logging Capture Strategy

BlissfulLogger uses `propagate=False` and a custom RichHandler, so `caplog` won't work. Instead:

```python
import logging
import unittest
from unittest.mock import patch
from io import StringIO

class TestCaptionDirectory(unittest.TestCase):
    def setUp(self):
        # Capture warnings by adding a temporary handler
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.WARNING)

        # Get the actual logger instance
        from musubi_tuner.dataset.image_video_dataset import logger
        self.original_handlers = logger.logger.handlers.copy()
        logger.logger.addHandler(self.handler)

    def tearDown(self):
        from musubi_tuner.dataset.image_video_dataset import logger
        logger.logger.removeHandler(self.handler)

    def get_log_output(self) -> str:
        return self.log_capture.getvalue()

    def assertWarningContains(self, *substrings):
        output = self.get_log_output()
        for s in substrings:
            self.assertIn(s, output)
```

### Test Cases

#### 1. `test_backward_compat_no_caption_directory`
- **Setup:** tmpdir with `a.png` + `a.txt`
- **Action:** Create `ImageDirectoryDatasource(image_dir, caption_extension=".txt")`
- **Assert:** `len(datasource.image_paths) == 1`, `get_caption(0)` returns content from `a.txt`
- **Guards:** default behavior unchanged

#### 2. `test_separate_caption_directory`
- **Setup:** `/images/a.png`, `/captions/a.txt`
- **Action:** Create datasource with `caption_directory="/captions"`
- **Assert:** `a.png` included, caption read from `/captions/a.txt`
- **Guards:** new feature works

#### 3. `test_multi_dot_caption_extension`
- **Setup:** `a.png`, caption file `a.flux2.txt`, `caption_extension=".flux2.txt"`
- **Action:** Create datasource
- **Assert:** `a.png` included, caption loads correctly
- **Guards:** multi-dot extensions work by construction

#### 4. `test_multi_dot_media_name`
- **Setup:** `foo.bar.png`, caption `foo.bar.txt`
- **Action:** Create datasource
- **Assert:** `foo.bar.png` included, caption loads
- **Guards:** multi-dot media names work

#### 5. `test_warning_with_truncation`
- **Setup:** 30 images, only 5 have captions
- **Action:** Create datasource
- **Assert:**
  - `len(datasource.image_paths) == 5`
  - Warning contains `"25/30"`, `"caption_extension="`, `"caption_directory="`, `"and 5 more"`
  - Exactly one warning emitted
- **Guards:** UX + anti-spam

#### 6. `test_zero_matches_hard_error`
- **Setup:** images exist, no captions
- **Action:** Create datasource
- **Assert:** Raises `ValueError` with message containing caption dir/ext and total count
- **Guards:** prevents expensive runs on misconfig

#### 7. `test_zero_media_no_error`
- **Setup:** empty image dir, caption_extension set
- **Action:** Create datasource
- **Assert:** No error raised, `datasource.image_paths == []`, log contains "Found 0 images"
- **Guards:** don't misdiagnose empty media dir as caption problem

#### 8. `test_duplicate_basename_hard_error`
- **Setup:** `a.png` and `a.jpg` in same dir
- **Action:** Create datasource
- **Assert:** Raises `ValueError` with "Duplicate" and both file paths
- **Guards:** prevents cache corruption

#### 9. `test_video_caption_filtering`
- **Setup:** `v1.mp4`, `v2.mp4`; only `v1.txt` exists
- **Action:** Create `VideoDirectoryDatasource`
- **Assert:** `video_paths == [v1.mp4]`, warning about `v2`
- **Guards:** video bug fix

#### 10. `test_invalid_caption_directory`
- **Setup:** `caption_directory="/nonexistent"`
- **Action:** Create datasource
- **Assert:** Raises `ValueError` with "does not exist"
- **Guards:** fail-fast validation

#### 11. `test_caption_directory_is_file`
- **Setup:** `caption_directory` points to a file
- **Action:** Create datasource
- **Assert:** Raises `ValueError` with "not a directory"
- **Guards:** fail-fast validation

#### 12. `test_empty_caption_extension`
- **Setup:** `caption_extension=""`
- **Action:** Create datasource
- **Assert:** Raises `ValueError` with "empty"
- **Guards:** fail-fast validation

#### 13. `test_whitespace_caption_extension_stripped`
- **Setup:** `caption_extension=" .txt "`
- **Action:** Create datasource with valid image/caption
- **Assert:** Warning about stripping, caption still loads
- **Guards:** handles common typo

#### 14. `test_caption_extension_no_dot_warning`
- **Setup:** `caption_extension="txt"` (missing dot)
- **Action:** Create datasource
- **Assert:** Warning about missing dot
- **Guards:** catches common mistake

---

## Documentation Updates (`docs/dataset_config.md`)

### New Subsection: `caption_directory`

```markdown
### `caption_directory` (optional)

Directory containing caption files. Defaults to `image_directory` or `video_directory`.

**Only applies to directory-based datasets** (not JSONL).

#### Caption Matching Rule

Caption filename must be: `<media_basename_without_extension><caption_extension>`

Examples:
- `portrait.png` + `caption_extension=".txt"` → expects `portrait.txt`
- `portrait.png` + `caption_extension=".flux2.txt"` → expects `portrait.flux2.txt`
- `foo.bar.png` + `caption_extension=".txt"` → expects `foo.bar.txt` (multi-dot media names work)

#### Example: Shared Images, Different Captions per Model

```toml
# FLUX.2 with FLUX-style prompts
[[datasets]]
image_directory = "/data/OLVA/images"
caption_directory = "/data/OLVA/flux2_captions"
cache_directory = "/data/OLVA/flux2_cache"
caption_extension = ".txt"

# WAN 2.2 with WAN-style prompts (same images!)
[[datasets]]
image_directory = "/data/OLVA/images"
caption_directory = "/data/OLVA/wan_captions"
cache_directory = "/data/OLVA/wan_cache"
caption_extension = ".txt"
```

**Important:** Each dataset must have a unique `cache_directory`. Using the same images
with different captions requires different cache directories to avoid cache collisions.

#### Behavior

- **Images/videos without matching captions are filtered out** with a warning
- **If ALL items are filtered (0 matches):** Hard error with actionable message
- **Duplicate basenames:** Hard error (e.g., `photo.jpg` + `photo.png` causes cache collisions)

#### Troubleshooting

**"caption_directory does not exist" error:**
- The specified path doesn't exist. Check for typos. Use absolute paths for reliability.

**"No images/videos with matching caption files found" error:**
- All media files were filtered because none had matching captions
- Check that `caption_directory` contains files with the correct `caption_extension`
- Verify caption filenames match media basenames exactly

**"Duplicate basenames detected" error:**
- Multiple images share the same base name (e.g., `photo.jpg` and `photo.png`)
- This would cause cache file collisions (TE caches use basename only)
- Fix: Rename files to have unique basenames, or use `image_jsonl_file` for explicit paths

**Stale TE caches after changing captions:**
- Changing caption text requires recaching text encoder outputs
- Recommended: use a separate `cache_directory` per caption strategy/model

**Caption not found at runtime despite passing filter:**
- Possible case sensitivity mismatch (Linux is case-sensitive; macOS/Windows often aren't)
- Possible Unicode normalization issue (different tools produce visually-identical but binary-different filenames)
- Try renaming files or normalizing Unicode

**"caption_extension does not start with '.'" warning:**
- You likely meant `.txt` but wrote `txt`
- Without the dot, it expects files like `footxt` instead of `foo.txt`

#### Platform Notes

- **Case sensitivity:** Linux is case-sensitive; macOS/Windows are often case-insensitive.
  Captions work on macOS but fail on Linux if case doesn't match exactly.
- **Path resolution:** Relative paths are resolved relative to the current working directory (CWD),
  not the config file location. Use absolute paths for reliability.
- **Non-recursive:** Only files directly in the directory are scanned (no subdirectories).
```

---

## Implementation Order

1. **Helper functions** - Add `_filter_paths_by_caption()`, `_check_duplicate_basenames()` to `image_video_dataset.py`
2. **Config layer** - Add fields/schemas to `config_utils.py`
3. **Datasources** - Update `ImageDirectoryDatasource`, `VideoDirectoryDatasource`
4. **Dataset classes** - Thread `caption_directory` through constructors
5. **Config logging** - Add `caption_directory` to dataset info output
6. **Tests** - Create `tests/test_dataset_caption_directory.py`
7. **Documentation** - Update `docs/dataset_config.md`

---

## Behavior Changes Summary

| Change | Before | After |
|--------|--------|-------|
| Image caption filtering | Silent | Warning + filter |
| Video caption filtering | Crash at runtime (FileNotFoundError) | Warning + filter (same as images) |
| Duplicate basenames | Silent (causes cache corruption) | Hard error |
| Empty media directory with caption_extension | "found 0 images" log | Same (no change) |
| All items filtered | Silent empty dataset | Hard error |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaks existing configs | Default to current behavior when `caption_directory` not set |
| Misdiagnose empty media dir as caption problem | Three-way behavior: `total_count==0` is not a caption error |
| Silent cache corruption from duplicate basenames | Hard error on duplicates |
| Multi-dot extension bug | Avoided by construction (never parse caption filenames) |
| Inconsistent basename derivation | Same `splitext(basename)` rule everywhere |
| Misleading errors from wrong validation order | Validate dirs before filtering |
| Test flakiness from BlissfulLogger | Custom handler attachment strategy |

---

## Not In Scope (Future Enhancements)

- `text_encoder_cache_directory` - Separate TE cache location
- `require_caption` config flag - Strict mode
- GUI updates - Lower priority
- Recursive directory scanning
- Case-insensitive caption matching
- More robust cache key scheme
