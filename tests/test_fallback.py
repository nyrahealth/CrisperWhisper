"""Unit tests for the coverage-gated temperature fallback (fallback.py).

Pure-logic tests -- no model, no GPU.  A tiny stub engine stands in for the
real backends: it only needs ``decode_tokens`` (word counting), ``generate``
(sibling reference decode) and ``generate_sampled`` (recovery draws).
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.fallback import (
    DISCREPANCY_RATIO,
    MIN_REF_WORDS,
    _discrepant,
    decode_with_coverage_fallback,
    is_undercovered,
    speech_active_seconds,
    word_count,
)


def _speechy_mel(active_frames: int = 2000, total_frames: int = 3000) -> np.ndarray:
    """A synthetic log-mel with a clearly speech-active leading region."""
    mel = np.full((128, total_frames), -1.0, dtype=np.float32)
    mel[:, :active_frames] = 1.0
    return mel


class _StubEngine:
    """Minimal engine stub: every token id > 0 decodes to one word.

    Distinct ids matter -- the fallback rejects sampled decodes that trip
    the repetition-loop detector, so "good" outputs must be non-repetitive.
    """

    def __init__(self, sampled_outputs=None):
        # queue of outputs returned by successive generate_sampled calls
        self._sampled = list(sampled_outputs or [])
        self.sampled_calls = 0
        self.all_special_ids: set[int] = set()
        self.default_suppress_tokens: list[int] = []

    def decode_tokens(self, ids, skip_special=True):
        return " ".join(f"w{int(t)}" for t in ids if int(t) > 0)

    def generate(self, features, prompts, max_length=256, suppress_tokens=None):
        # Sibling reference decode: plenty of (distinct) words.
        return [list(range(1, 21))]

    def generate_sampled(self, features, prompt_tokens, *, max_length,
                         temperature, topk, seed, suppress_tokens=None):
        self.sampled_calls += 1
        if self._sampled:
            return self._sampled.pop(0)
        return list(range(1, 21))

    def cross_attention_for_tokens(self, features, prompt_tokens, gen_ids):
        return np.ones((len(gen_ids), 10), dtype=np.float32)


class _NoSamplingEngine:
    """Engine without generate_sampled (like SpeculativeDecoder)."""

    def decode_tokens(self, ids, skip_special=True):
        return ""


class TestSpeechActiveSeconds:
    def test_none_mel(self):
        assert speech_active_seconds(None) is None

    def test_flat_mel_is_silent(self):
        mel = np.zeros((128, 3000), dtype=np.float32)
        assert speech_active_seconds(mel) == 0.0

    def test_active_region_counted(self):
        sec = speech_active_seconds(_speechy_mel(active_frames=2000))
        # 2000 frames * 10ms = 20s of active audio.
        assert sec == pytest.approx(20.0, abs=1.0)

    def test_empty_mel(self):
        assert speech_active_seconds(np.zeros((128, 0), dtype=np.float32)) == 0.0


class TestIsUndercovered:
    def test_dense_speech_few_words(self):
        assert is_undercovered(2, _speechy_mel()) is True

    def test_dense_speech_enough_words(self):
        assert is_undercovered(60, _speechy_mel()) is False

    def test_short_speech_never_triggers(self):
        # Under the 5s speech minimum: not eligible regardless of word count.
        assert is_undercovered(0, _speechy_mel(active_frames=100)) is False

    def test_no_mel(self):
        assert is_undercovered(0, None) is False


class TestDiscrepant:
    def test_collapse(self):
        assert _discrepant(3, 20) is True

    def test_ratio_boundary(self):
        ref = 20
        just_below = int(DISCREPANCY_RATIO * ref) - 1
        assert _discrepant(just_below, ref) is True
        assert _discrepant(ref, ref) is False

    def test_tiny_reference_ignored(self):
        assert _discrepant(0, MIN_REF_WORDS - 1) is False


class TestWordCount:
    def test_counts_decoded_words(self):
        eng = _StubEngine()
        assert word_count(eng, [1, 1, 0, 1]) == 3
        assert word_count(eng, []) == 0


class TestDecodeWithCoverageFallback:
    def test_engine_without_sampling_passthrough(self):
        gen = [1, 1]
        ids, attn = decode_with_coverage_fallback(
            _NoSamplingEngine(), None, _speechy_mel(), [0], gen, None,
            max_length=64, want_attention=False, ref_word_count=50,
        )
        assert ids is gen and attn is None

    def test_disabled_passthrough(self):
        eng = _StubEngine()
        gen = [1]
        ids, _ = decode_with_coverage_fallback(
            eng, None, _speechy_mel(), [0], gen, None,
            max_length=64, want_attention=False, enabled=False,
            ref_word_count=50,
        )
        assert ids is gen
        assert eng.sampled_calls == 0

    def test_not_collapsed_no_recovery(self):
        eng = _StubEngine()
        gen = [1] * 30
        ids, _ = decode_with_coverage_fallback(
            eng, None, _speechy_mel(), [0], gen, None,
            max_length=64, want_attention=False, ref_word_count=30,
        )
        assert ids is gen
        assert eng.sampled_calls == 0

    def test_dual_discrepancy_triggers_recovery(self):
        recovered = list(range(1, 26))
        eng = _StubEngine(sampled_outputs=[recovered])
        collapsed = [1, 1]  # 2 words vs sibling's 30 -> collapse
        ids, _ = decode_with_coverage_fallback(
            eng, None, _speechy_mel(), [0], collapsed, None,
            max_length=64, want_attention=False, ref_word_count=30,
        )
        assert ids == recovered
        assert eng.sampled_calls >= 1

    def test_recovery_attention_recomputed(self):
        recovered = list(range(1, 26))
        eng = _StubEngine(sampled_outputs=[recovered])
        old_attn = np.zeros((2, 10), dtype=np.float32)
        ids, attn = decode_with_coverage_fallback(
            eng, None, _speechy_mel(), [0], [1, 1], old_attn,
            max_length=64, want_attention=True, ref_word_count=30,
        )
        assert ids == recovered
        assert attn is not None and attn.shape[0] == len(recovered)

    def test_looping_samples_rejected(self):
        # A sampled decode that is one long repetition loop must be rejected;
        # with no clean sample available the base decode is kept.
        looping = [7] * 40
        eng = _StubEngine(sampled_outputs=[looping] * 12)
        base = [1, 1]
        ids, _ = decode_with_coverage_fallback(
            eng, None, _speechy_mel(), [0], base, None,
            max_length=64, want_attention=False, ref_word_count=30,
        )
        assert ids == base
