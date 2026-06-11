import json

import pytest

from musubi_tuner.ideogram4 import caption_verifier as cv_module
from musubi_tuner.ideogram4.caption_verifier import CaptionVerifier, reset_plain_text_notice, verify_caption


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


# --- Plain-text flood suppression (backlog P0-3) ---------------------------------------------------


@pytest.fixture
def rearm_notice():
    reset_plain_text_notice()
    yield
    reset_plain_text_notice()


def test_plain_text_notice_emitted_once_then_suppressed(rearm_notice, monkeypatch):
    infos, warnings = [], []
    monkeypatch.setattr(cv_module._caption_logger, "info", lambda msg, *a, **k: infos.append(msg))
    monkeypatch.setattr(cv_module._caption_logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    for _ in range(5):
        issues = verify_caption("a photo of a cat")
        assert issues  # return value unaffected by suppression

    assert len(infos) == 1 and "plain text" in infos[0]
    assert warnings == []


def test_malformed_json_still_warns_per_caption(rearm_notice, monkeypatch):
    # A caption that LOOKS like attempted JSON (leading '{') but fails to parse is actionable —
    # it must keep the per-caption WARNING, not be mistaken for plain text.
    infos, warnings = [], []
    monkeypatch.setattr(cv_module._caption_logger, "info", lambda msg, *a, **k: infos.append(msg))
    monkeypatch.setattr(cv_module._caption_logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    for _ in range(3):
        verify_caption('{"high_level_description": broken')

    assert len(warnings) == 3
    assert infos == []


def test_schema_issues_on_valid_json_still_warn(rearm_notice, monkeypatch):
    # Valid JSON with schema problems is not plain text; warnings stay per-caption.
    warnings = []
    monkeypatch.setattr(cv_module._caption_logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    verify_caption(json.dumps({"unknown_key": 1}))
    assert len(warnings) == 1 and "unknown" in warnings[0]


def test_strict_mode_unaffected_by_suppression(rearm_notice):
    verify_caption("a photo of a cat")  # consumes the one-shot notice
    with pytest.raises(ValueError, match="caption issues"):
        verify_caption("another plain caption", strict=True)
