"""Default model id, size-shorthand resolution, and hotword guarding."""

import inspect
import warnings

import pytest

from crisperwhisper.model import (
    DEFAULT_MODEL,
    OFFICIAL_MODELS,
    CrisperWhisperModel,
    resolve_model_id,
)


def test_default_is_large():
    assert DEFAULT_MODEL == "nyralabs/CrisperWhisper2.0_large"
    sig = inspect.signature(CrisperWhisperModel.__init__)
    assert sig.parameters["model_name_or_path"].default == DEFAULT_MODEL


def test_official_sizes():
    assert set(OFFICIAL_MODELS) == {
        "large", "turbo", "medium", "small",
        "large_pro", "turbo_pro", "medium_pro", "small_pro",
    }
    for size, repo in OFFICIAL_MODELS.items():
        assert repo == f"nyralabs/CrisperWhisper2.0_{size}"


def test_shorthand_resolution():
    assert resolve_model_id("turbo") == "nyralabs/CrisperWhisper2.0_turbo"
    assert resolve_model_id("large_pro") == "nyralabs/CrisperWhisper2.0_large_pro"
    # Full ids and local paths pass through untouched.
    assert resolve_model_id("nyralabs/CrisperWhisper2.0_large") == (
        "nyralabs/CrisperWhisper2.0_large"
    )
    assert resolve_model_id("/models/my-finetune") == "/models/my-finetune"
    assert resolve_model_id("nyrahealth/CrisperWhisper") == "nyrahealth/CrisperWhisper"


def _bare_model(model_path: str, version: int = 2) -> CrisperWhisperModel:
    """An uninitialized instance, enough to exercise the hotword guard."""
    m = CrisperWhisperModel.__new__(CrisperWhisperModel)
    m._model_path = model_path
    m._model_version = version
    return m


def test_hotwords_warn_on_standard_model():
    m = _bare_model("nyralabs/CrisperWhisper2.0_large")
    with pytest.warns(UserWarning, match="Pro model"):
        m._warn_if_hotwords_unsupported(["Nyra"])


def test_hotwords_no_warning_when_supported_or_absent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Pro model: allowed.
        _bare_model("nyralabs/CrisperWhisper2.0_large_pro")._warn_if_hotwords_unsupported(
            ["Nyra"]
        )
        # No hotwords passed: nothing to warn about.
        _bare_model("nyralabs/CrisperWhisper2.0_large")._warn_if_hotwords_unsupported(None)
        _bare_model("nyralabs/CrisperWhisper2.0_large")._warn_if_hotwords_unsupported([])
        # v1 has its own feature warning; the Pro guard stays silent.
        _bare_model("nyrahealth/CrisperWhisper", version=1)._warn_if_hotwords_unsupported(
            ["Nyra"]
        )
