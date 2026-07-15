"""Default model id and size-shorthand resolution."""

import inspect

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
    # Full ids and local paths pass through untouched.
    assert resolve_model_id("nyralabs/CrisperWhisper2.0_large") == (
        "nyralabs/CrisperWhisper2.0_large"
    )
    assert resolve_model_id("/models/my-finetune") == "/models/my-finetune"
    assert resolve_model_id("nyrahealth/CrisperWhisper") == "nyrahealth/CrisperWhisper"
