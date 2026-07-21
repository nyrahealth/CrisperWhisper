"""Suppress-token semantics across both backends.

Logic tests (no model): normalization of the config list, the ``-1``
sentinel, and the transformers backend passing an explicit (even empty)
list to HuggingFace ``generate`` instead of falling back to the model's
generation_config.

End-to-end tests (env-gated): suppressing a token that greedy decoding
emitted must remove it from the output -- on the ct2 backend
(``@pytest.mark.gpu``) and the transformers backend
(``@pytest.mark.transformers``).
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.model import _sanitize_suppress_tokens


class TestSanitizeSuppressTokens:
    def test_none_keeps_default(self):
        assert _sanitize_suppress_tokens(None) is None

    def test_empty_stays_empty(self):
        assert _sanitize_suppress_tokens([]) == []

    def test_negatives_dropped(self):
        assert _sanitize_suppress_tokens([-1, 220, -5, 50257]) == [220, 50257]


class TestCT2Normalization:
    def test_clean_suppress_tokens_helper(self):
        pytest.importorskip("ctranslate2")
        from crisperwhisper.engine import _clean_suppress_tokens

        assert _clean_suppress_tokens([-1, 0, 220]) == [0, 220]
        assert _clean_suppress_tokens([]) == []


class TestTransformersSuppressKwargs:
    """The TF backend must always pass an explicit suppress list to
    HF ``generate`` -- ``[]`` disables suppression rather than silently
    falling back to the model's generation_config (empty-list parity with
    the ct2 backend)."""

    def _dummy_engine(self, captured: dict):
        torch = pytest.importorskip("torch")
        from crisperwhisper.transformers_engine import TransformersEngine

        eng = object.__new__(TransformersEngine)
        eng.device = torch.device("cpu")
        eng.default_suppress_tokens = [5, 7]

        class _FakeModel:
            def generate(self, features, **kwargs):
                captured.update(kwargs)
                prefix = kwargs["decoder_input_ids"][0].tolist()
                return torch.tensor([prefix + [42]])

        eng.model = _FakeModel()
        return eng

    def test_default_list_passed(self):
        captured: dict = {}
        eng = self._dummy_engine(captured)
        out = eng._run_generate(None, [1, 2], 4)
        assert captured["suppress_tokens"] == [5, 7]
        assert out == [42]

    def test_explicit_list_overrides(self):
        captured: dict = {}
        eng = self._dummy_engine(captured)
        eng._run_generate(None, [1, 2], 4, suppress_tokens=[9])
        assert captured["suppress_tokens"] == [9]

    def test_empty_list_disables(self):
        captured: dict = {}
        eng = self._dummy_engine(captured)
        eng._run_generate(None, [1, 2], 4, suppress_tokens=[])
        assert captured["suppress_tokens"] == []

    def test_negatives_filtered(self):
        captured: dict = {}
        eng = self._dummy_engine(captured)
        eng._run_generate(None, [1, 2], 4, suppress_tokens=[-1, 3])
        assert captured["suppress_tokens"] == [3]


# ---------------------------------------------------------------------------
# End-to-end: suppressing an emitted token removes it (both backends).
# ---------------------------------------------------------------------------

def _first_content_token(engine, gen_ids) -> int | None:
    for t in gen_ids:
        if int(t) not in engine.all_special_ids:
            return int(t)
    return None


def _load_audio_array(path: str) -> np.ndarray:
    from crisperwhisper.audio import load_audio

    return load_audio(path)


def _suppress_roundtrip(engine, audio: np.ndarray):
    """Greedy-decode, then re-decode with the first content token suppressed."""
    from crisperwhisper.prompt import PromptBuilder

    prompt = PromptBuilder(engine).verbatim()
    features = engine.extract_features(audio)
    base = engine.generate(features, [prompt], max_length=64)[0]
    target = _first_content_token(engine, base)
    assert target is not None, "baseline decode produced no content tokens"

    suppressed = engine.generate(
        features, [prompt], max_length=64, suppress_tokens=[target],
    )[0]
    assert target not in [int(t) for t in suppressed]

    # Explicit empty list must not crash and must still decode.
    unsuppressed = engine.generate(
        features, [prompt], max_length=64, suppress_tokens=[],
    )[0]
    assert len(unsuppressed) > 0


@pytest.mark.gpu
def test_ct2_suppress_tokens_respected(model, en_audio):
    _suppress_roundtrip(model._engine, _load_audio_array(en_audio))


@pytest.mark.gpu
def test_ct2_model_level_suppress(model, en_audio):
    """transcribe() accepts suppress_tokens and still produces text."""
    res = model.transcribe(en_audio, suppress_tokens=[])
    assert res.text
    res2 = model.transcribe(en_audio)
    assert res2.text


@pytest.mark.transformers
def test_transformers_suppress_tokens_respected(tf_engine, en_audio):
    _suppress_roundtrip(tf_engine, _load_audio_array(en_audio))


@pytest.mark.transformers
def test_transformers_model_level_suppress(tf_model, en_audio):
    res = tf_model.transcribe(en_audio, suppress_tokens=[])
    assert res.text


@pytest.mark.transformers
def test_transformers_begin_suppress_cleared(tf_engine):
    """First-token suppression is disabled to match the ct2 backend."""
    assert tf_engine.model.generation_config.begin_suppress_tokens is None
