import json

import pytest

from musubi_tuner.ideogram4.caption_verifier import CaptionVerifier, verify_caption


def test_plain_text_caption_warns_but_does_not_raise():
    # H4: a plain .txt LoRA caption parses as invalid JSON. Default policy is WARN-ONLY — must not raise.
    issues = verify_caption("a photo of a cat")
    assert issues and any("invalid JSON" in i for i in issues)


def test_plain_text_caption_raises_only_when_strict():
    with pytest.raises(ValueError, match="caption issues"):
        verify_caption("a photo of a cat", strict=True)


def test_structured_json_caption_parses_unlike_plain_text():
    # A structured caption parses as JSON, so it is NOT flagged "invalid JSON" the way plain text is.
    # (It may still warn on the detailed schema — that's the verifier's job, not what we assert here.)
    caption = json.dumps(
        {"high_level_description": "a lake", "compositional_deconstruction": {"background": "hills", "elements": []}},
        ensure_ascii=False,
    )
    issues = verify_caption(caption)
    assert not any("invalid JSON" in i for i in issues)


def test_verifier_returns_list_and_never_raises_directly():
    # The underlying verifier itself only RETURNS issues (policy lives in verify_caption).
    assert isinstance(CaptionVerifier().verify_raw("not json"), list)
