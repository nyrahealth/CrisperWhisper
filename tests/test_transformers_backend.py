"""Tests for the HuggingFace Transformers inference backend.

Pure-logic tests (backend selection) run anywhere.  The end-to-end tests
are marked ``transformers`` and skip unless ``CW2_HF_MODEL_PATH`` points at
a CrisperWhisper v2 HuggingFace checkpoint and ``torch`` is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Backend selection logic (no model / torch needed).
# ---------------------------------------------------------------------------

class TestBackendResolution:
    def test_explicit_backends_pass_through(self):
        from crisperwhisper.model import CrisperWhisperModel

        assert CrisperWhisperModel._resolve_backend("ct2", 2) == "ct2"
        assert (
            CrisperWhisperModel._resolve_backend("transformers", 2)
            == "transformers"
        )

    def test_v1_always_transformers(self):
        from crisperwhisper.model import CrisperWhisperModel

        # v1 models always use the legacy HF pipeline regardless of request.
        assert CrisperWhisperModel._resolve_backend("ct2", 1) == "transformers"
        assert CrisperWhisperModel._resolve_backend("auto", 1) == "transformers"

    def test_auto_prefers_available_backend(self):
        from crisperwhisper.model import CrisperWhisperModel, _backend_available

        resolved = CrisperWhisperModel._resolve_backend("auto", 2)
        if _backend_available("ct2"):
            assert resolved == "ct2"
        elif _backend_available("transformers"):
            assert resolved == "transformers"

    def test_invalid_backend_raises(self):
        from crisperwhisper.model import CrisperWhisperModel

        with pytest.raises(ValueError):
            CrisperWhisperModel._resolve_backend("jax", 2)

    def test_require_backend_message(self, monkeypatch):
        from crisperwhisper import model as model_mod

        monkeypatch.setattr(model_mod, "_backend_available", lambda b: False)
        with pytest.raises(ImportError, match=r"crisperwhisper\[transformers\]"):
            model_mod._require_backend("transformers")


class TestFirstStepBan:
    def test_bans_only_on_first_step(self):
        import numpy as _np

        from crisperwhisper.transformers_engine import _FirstStepBan

        ban = _FirstStepBan(prefix_len=3, banned=[5])

        # At the first generated step (cur len == prefix len) the token is
        # masked; on later steps it is left untouched.
        scores = _np.zeros((1, 10), dtype=_np.float32)
        ids_first = _np.zeros((1, 3), dtype=_np.int64)
        out = ban(ids_first, scores.copy())
        assert out[0, 5] == float("-inf")

        ids_later = _np.zeros((1, 4), dtype=_np.int64)
        out2 = ban(ids_later, scores.copy())
        assert out2[0, 5] == 0.0


# ---------------------------------------------------------------------------
# End-to-end transformers backend (needs CW2_HF_MODEL_PATH + torch).
# ---------------------------------------------------------------------------

@pytest.mark.transformers
class TestTransformersEndToEnd:
    def test_short_form_text(self, tf_model, en_audio):
        result = tf_model.transcribe(en_audio)
        assert tf_model.backend == "transformers"
        assert result.text.strip()
        assert result.words is None

    def test_word_timestamps_monotonic(self, tf_model, en_audio):
        result = tf_model.transcribe(en_audio, word_timestamps=True)
        words = result.words
        assert words and len(words) > 5
        for i in range(1, len(words)):
            assert words[i].start >= words[i - 1].start - 1e-6
        assert all(w.end >= w.start for w in words)
        # timestamps stay within the audio (+ small slack)
        assert words[-1].end <= result.duration + 1.0

    def test_longform_monotonic_seams(self, tf_model, en_audio):
        audio, sr = sf.read(en_audio)
        audio = audio.astype(np.float32)
        long_audio = np.tile(audio, 3)  # > 30s -> multi-chunk longform
        result = tf_model.transcribe(long_audio, sr=sr, word_timestamps=True)
        assert result.chunks and len(result.chunks) >= 2
        words = result.words
        assert words and len(words) > 10
        for i in range(1, len(words)):
            assert words[i].start >= words[i - 1].start - 1e-6

    def test_speculative_request_warns_and_runs(self, tf_model, en_audio):
        with pytest.warns(UserWarning, match="Speculative decoding is not"):
            result = tf_model.transcribe(en_audio, speculative_decoding=True)
        assert result.text.strip()


@pytest.mark.transformers
class TestTransformersRepair:
    def test_repair_removes_loop(self, tf_engine, en_audio):
        from crisperwhisper.hallucination import (
            DEFAULT_REPAIR_THRESHOLDS,
            find_token_loop,
        )
        from crisperwhisper.prompt import PromptBuilder

        audio, _ = sf.read(en_audio)
        audio = audio.astype(np.float32)
        prompt = PromptBuilder(tf_engine, language="en").verbatim()
        features = tf_engine.extract_features(audio)

        repaired = tf_engine.generate_with_repair(
            features, prompt, max_length=220, hallucination_mitigation=True,
        )
        # The user-facing guarantee: the repaired output never contains a
        # detectable consecutive n-gram loop at the default thresholds.
        assert find_token_loop(repaired, reps=DEFAULT_REPAIR_THRESHOLDS) is None

    def test_attention_rows_match_token_count(self, tf_engine, en_audio):
        from crisperwhisper.prompt import PromptBuilder

        audio, _ = sf.read(en_audio)
        audio = audio.astype(np.float32)
        prompt = PromptBuilder(tf_engine, language="en").verbatim()
        tf_engine.enable_attention(None)
        features = tf_engine.extract_features(audio)
        gen_ids, attention = tf_engine.generate_with_repair_and_attention(
            features, prompt, max_length=180,
        )
        assert attention.shape[0] == len(gen_ids)
        assert attention.shape[1] > 0  # F_enc encoder frames


@pytest.mark.transformers
class TestTransformersTokenizerParity:
    def test_word_grouping_uses_leading_space(self, tf_engine):
        from crisperwhisper.word_timing import group_tokens_into_words

        ids = tf_engine.encode_text(" hello world")
        pieces = [tf_engine.tokenizer.decode([t]) for t in ids]
        _, words = group_tokens_into_words(ids, pieces)
        assert words == ["hello", "world"]
