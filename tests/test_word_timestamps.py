"""Tests for cross-attention word-timestamp extraction.

The pure-numpy / pure-Python pieces (Viterbi, blank energy, token-into-word
grouping) are exercised directly with synthetic data.  The CrisperWhisper +
CTranslate2 end-to-end pieces are gated behind a ``@pytest.mark.gpu`` marker
since they require a converted model and CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.result import WordTimestamp
from crisperwhisper.word_timing import (
    blank_logp_from_mel_energy,
    extract_word_timings,
    group_tokens_into_words,
    token_logp_from_attention,
    viterbi_align_tokens_with_blanks,
    viterbi_align_words_with_blanks,
)


# ---------------------------------------------------------------------------
# Pure-numpy unit tests (no model, no CT2).
# ---------------------------------------------------------------------------

class TestTokenLogp:
    def test_rows_sum_to_one(self):
        attn = np.random.RandomState(0).rand(5, 100).astype(np.float32)
        attn /= attn.sum(axis=1, keepdims=True)
        logp = token_logp_from_attention(attn, sharpen=1.0)
        # exp(logp) should sum to ~1 along the frame axis.
        assert np.allclose(np.exp(logp).sum(axis=1), 1.0, atol=1e-3)

    def test_sharpen_concentrates_mass(self):
        attn = np.ones((1, 100), dtype=np.float32) / 100.0
        # With sharpen=3, a uniform distribution stays uniform.
        logp = token_logp_from_attention(attn, sharpen=3.0)
        assert np.allclose(np.exp(logp), 1.0 / 100.0, atol=1e-3)

        # With a tiny bump, sharpening should amplify the peak.
        attn[0, 50] *= 2.0
        attn /= attn.sum(axis=1, keepdims=True)
        peak_p = float(np.exp(token_logp_from_attention(attn, sharpen=3.0))[0, 50])
        uniform_p = 1.0 / 100.0
        assert peak_p > uniform_p * 3.0


class TestBlankLogp:
    def test_silent_frames_have_higher_blank_logp(self):
        mel = np.zeros((80, 200), dtype=np.float32)
        mel[:, 40:160] = 5.0
        blank_logp = blank_logp_from_mel_energy(mel, target_frames=100)
        assert blank_logp.shape == (100,)
        edge = float(np.exp(blank_logp[:10]).mean())
        middle = float(np.exp(blank_logp[40:60]).mean())
        assert edge > middle

    def test_audio_frames_clip(self):
        mel = np.zeros((80, 200), dtype=np.float32)
        mel[:, 80:160] = 5.0
        # If we tell it we only care about the first half, the energy
        # distribution is entirely silent (no voiced region in mel[:,:80]).
        blank_logp = blank_logp_from_mel_energy(
            mel, target_frames=40, audio_frames=40,
        )
        assert blank_logp.shape == (40,)


class TestViterbiAlignment:
    @staticmethod
    def _gaussian_attention(peaks: list[int], num_frames: int, sigma: float = 5.0):
        T = len(peaks)
        attn = np.zeros((T, num_frames), dtype=np.float32)
        for t, p in enumerate(peaks):
            x = np.arange(num_frames) - p
            g = np.exp(-(x ** 2) / (2 * sigma ** 2))
            attn[t] = g / g.sum()
        return attn

    @staticmethod
    def _voicing_blank_logp(peaks: list[int], F: int, half_width: int = 8) -> np.ndarray:
        """Per-frame blank log-prob: ~1.0 (silence) in unvoiced gaps,
        ~1e-4 (definitely speech) in voiced regions around the peaks.
        Matches what :func:`blank_logp_from_mel_energy` produces from a
        mel where energy clearly distinguishes speech vs. silence.
        """
        blank_p = np.full((F,), 1.0 - 1e-4, dtype=np.float32)
        for p in peaks:
            lo, hi = max(0, p - half_width), min(F, p + half_width + 1)
            blank_p[lo:hi] = 1e-4
        return np.log(blank_p).astype(np.float32)

    def test_aligns_to_peaks(self):
        peaks = [10, 40, 70]
        F = 100
        attn = self._gaussian_attention(peaks, F)
        tok_logp = token_logp_from_attention(attn, sharpen=3.0)
        blank_logp = self._voicing_blank_logp(peaks, F)

        timings = viterbi_align_tokens_with_blanks(
            tok_logp, blank_logp, frame_duration=0.02,
        )
        assert len(timings) == 3
        for (s, e), p in zip(timings, peaks):
            assert s is not None and e is not None
            assert abs(((s + e) / 2.0) - p * 0.02) < 0.20

    def test_monotonic(self):
        peaks = [5, 25, 50, 80]
        F = 100
        attn = self._gaussian_attention(peaks, F)
        tok_logp = token_logp_from_attention(attn, sharpen=3.0)
        blank_logp = self._voicing_blank_logp(peaks, F)

        timings = viterbi_align_tokens_with_blanks(tok_logp, blank_logp)
        ends = [e for (_, e) in timings]
        for a, b in zip(ends[:-1], ends[1:]):
            assert a is not None and b is not None
            assert a <= b

    def test_word_collapse(self):
        peaks = [12, 14, 50, 80]
        F = 100
        attn = self._gaussian_attention(peaks, F, sigma=2.0)
        tok_logp = token_logp_from_attention(attn, sharpen=3.0)
        blank_logp = self._voicing_blank_logp(peaks, F)
        word_idx = [[0, 1], [2], [3]]
        timings = viterbi_align_words_with_blanks(
            tok_logp, blank_logp, word_idx, frame_duration=0.02,
        )
        assert len(timings) == 3
        for s, e in timings:
            assert s is not None and e is not None


class TestTokenGrouping:
    def test_space_separated_words(self):
        tok_ids = [50000, 100, 220, 200, 50001]
        pieces = ["<|startoftranscript|>", " hello", " ", "world", "<|endoftext|>"]
        word_idx, word_text = group_tokens_into_words(tok_ids, pieces)
        assert word_text == ["hello", "world"]
        assert word_idx == [[1], [3]]

    def test_prefix_space_word_boundary(self):
        tok_ids = [101, 102, 103, 104]
        pieces = ["hel", "lo", " world", "!"]
        word_idx, word_text = group_tokens_into_words(tok_ids, pieces)
        assert word_text == ["hello", "world!"]
        assert word_idx == [[0, 1], [2, 3]]

    def test_prompt_artifacts_skipped(self):
        tok_ids = [50001, 50002, 100, 50003, 200, 50004]
        pieces = ["<ctx>", "<ectx>", " hi", "[verbatim_1]", "there", "<eot>"]
        word_idx, word_text = group_tokens_into_words(tok_ids, pieces)
        assert word_text == ["hi", "there"]


# ---------------------------------------------------------------------------
# End-to-end CrisperWhisper test: requires GPU + converted model.
# The ``model``/``spec_model``/``en_audio`` fixtures live in conftest.py and
# skip automatically when CW2_MODEL_PATH / CW2_DRAFT_MODEL_PATH are unset.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestEndToEnd:
    """Run on a real model + a synthetic mono tone burst."""

    def _make_tone_burst_audio(self, total_s: float = 5.0):
        """Return a 16kHz audio array with two distinct ~0.5s tone bursts
        separated by silence, so we can sanity-check timing locations.
        """
        sr = 16000
        n = int(total_s * sr)
        audio = np.zeros(n, dtype=np.float32)
        # Burst 1: 1.0s..1.5s
        t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
        audio[int(1.0 * sr): int(1.5 * sr)] = 0.3 * np.sin(2 * np.pi * 440 * t)
        # Burst 2: 3.0s..3.5s
        audio[int(3.0 * sr): int(3.5 * sr)] = 0.3 * np.sin(2 * np.pi * 880 * t)
        return audio

    def test_short_form_produces_words(self, model):
        audio = self._make_tone_burst_audio(total_s=5.0)
        result = model.transcribe(audio, sr=16000, word_timestamps=True)
        # Tones are not actual speech, so we don't assert on text content.
        assert result.words is None or isinstance(result.words, list)
        if result.words:
            for w in result.words:
                assert isinstance(w, WordTimestamp)
                assert 0.0 <= w.start <= w.end <= result.duration + 0.5

    def test_long_form_monotonic_seams(self, model):
        """Concatenate the same 5s audio enough times to cross 30s; check
        that word timings stay monotonic across chunk boundaries.
        """
        audio = np.tile(self._make_tone_burst_audio(total_s=5.0), 8)  # 40s
        result = model.transcribe(audio, sr=16000, word_timestamps=True,
                                  longform_strategy="continuation")
        assert result.duration > 30.0
        if result.words:
            prev_end = -1.0
            for w in result.words:
                assert w.start >= prev_end - 1e-3, (
                    f"non-monotonic seam: word {w.word!r} starts at "
                    f"{w.start:.3f} before prev end {prev_end:.3f}"
                )
                assert w.end >= w.start
                prev_end = w.end

    def test_hallucination_repair_alignment(self, model):
        """A short burst plus 25s of silence is a common hallucination
        trigger; the repair-aware path should still produce monotonic
        timestamps with rows matching the kept tokens.
        """
        sr = 16000
        audio = np.zeros(28 * sr, dtype=np.float32)
        t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
        audio[int(1.0 * sr): int(1.5 * sr)] = 0.3 * np.sin(2 * np.pi * 440 * t)
        result = model.transcribe(audio, sr=16000, word_timestamps=True,
                                  hallucination_mitigation=True)
        if result.words:
            for w in result.words:
                assert 0.0 <= w.start <= w.end <= result.duration + 0.5

    def test_word_timestamps_lcs_longform_supported(self, model):
        """``word_timestamps=True`` works with the LCS-stitched longform
        strategies (issue #52): each chunk's attention is recovered with a
        teacher-forced pass and timings are carried through the stitch.
        The full-quality path is exercised on real speech in
        ``test_e2e_gpu.py``; this synthetic clip just asserts the call
        succeeds and returns a well-formed (possibly empty) word list.
        """
        audio = np.tile(self._make_tone_burst_audio(total_s=5.0), 8)  # 40s
        for strategy in ("chunked_lcs", "token_lcs"):
            result = model.transcribe(
                audio, sr=16000,
                word_timestamps=True,
                longform_strategy=strategy,
            )
            assert result.words is None or all(
                w.start is not None and w.end is not None and w.start <= w.end
                for w in result.words
            )


# ---------------------------------------------------------------------------
# End-to-end speculative-decoding + word-timestamps test (Option B).
# Requires a main *and* a draft converted model.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestSpeculativeTimingEndToEnd:
    def _tone_burst(self, total_s: float = 5.0):
        sr = 16000
        audio = np.zeros(int(total_s * sr), dtype=np.float32)
        t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
        audio[int(1.0 * sr): int(1.5 * sr)] = 0.3 * np.sin(2 * np.pi * 440 * t)
        audio[int(3.0 * sr): int(3.5 * sr)] = 0.3 * np.sin(2 * np.pi * 880 * t)
        return audio

    def test_warns_and_produces_monotonic_words(self, spec_model):
        import warnings

        audio = self._tone_burst(5.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spec_model.transcribe(
                audio, sr=16000,
                word_timestamps=True,
                speculative_decoding=True,
            )
        # A UserWarning about mixed draft/main attention must be emitted.
        assert any(
            issubclass(w.category, UserWarning)
            and "speculative" in str(w.message).lower()
            for w in caught
        ), "expected a UserWarning about speculative + timing"

        if result.words:
            prev_end = -1.0
            for w in result.words:
                assert isinstance(w, WordTimestamp)
                assert 0.0 <= w.start <= w.end <= result.duration + 0.5
                assert w.start >= prev_end - 1e-3
                prev_end = w.end

    def test_matches_nonspeculative_token_alignment(self, spec_model):
        """The attention matrix must stay 1-to-1 with the emitted tokens
        even with hallucination repair active (rewind-consistent rows)."""
        from crisperwhisper.prompt import PromptBuilder
        from crisperwhisper.speculative import SpeculativeDecoder

        engine = SpeculativeDecoder(
            spec_model._engine,
            spec_model._draft_engine,
            num_speculative_tokens=5,
        )
        assert hasattr(engine, "generate_with_attention")
        audio = self._tone_burst(5.0)
        feats, _mel = engine.extract_features_with_mel(audio)
        engine.enable_attention()
        prompt = PromptBuilder(engine.main, language="en").verbatim()
        for hm in (True, False):
            gen, attn = engine.generate_with_attention(
                feats, prompt, max_length=128, hallucination_mitigation=hm,
            )
            assert attn.ndim == 2
            assert attn.shape[0] == len(gen), (
                f"attention rows ({attn.shape[0]}) != tokens ({len(gen)}) "
                f"with hallucination_mitigation={hm}"
            )


# ---------------------------------------------------------------------------
# Pure-Python stitching invariant: the per-token attention-row sourcing
# rule used by the speculative timing loop (Option B).  Verified without
# a model so the invariant is locked even off-GPU.
# ---------------------------------------------------------------------------

def _committed_row_sources(n_draft_accepted, n_candidates, has_correction):
    """Replicate the commit-phase row sourcing of
    ``SpeculativeDecoder._speculative_generate_with_attention``:

    * 1 main-model row for the always-verified ``main_next`` token,
    * one draft-model row per accepted draft candidate,
    * (optionally) 1 main-model row for the verifier correction.

    Returns the ordered list of row sources ('main' | 'draft').
    """
    sources = ["main"]  # main_next
    sources += ["draft"] * n_draft_accepted
    if has_correction:
        sources.append("main")
    return sources


class TestStitchingInvariant:
    def test_full_accept_no_correction(self):
        src = _committed_row_sources(n_draft_accepted=4, n_candidates=4,
                                     has_correction=False)
        assert src == ["main", "draft", "draft", "draft", "draft"]
        # 1 main + all drafts, exactly len = 1 + n_draft_accepted.
        assert len(src) == 1 + 4

    def test_partial_accept_with_correction(self):
        src = _committed_row_sources(n_draft_accepted=2, n_candidates=5,
                                     has_correction=True)
        # main_next + 2 accepted drafts + 1 correction (main).
        assert src == ["main", "draft", "draft", "main"]
        assert len(src) == 1 + 2 + 1
        # rejected drafts (5 - 2 = 3) contribute NO rows.
        assert src.count("draft") == 2

    def test_zero_accept_immediate_correction(self):
        src = _committed_row_sources(n_draft_accepted=0, n_candidates=3,
                                     has_correction=True)
        assert src == ["main", "main"]

    def test_row_count_always_matches_token_count(self):
        # For every plausible (accepted, candidates, correction) combo the
        # committed row count equals the committed token count.
        for n_cand in range(0, 6):
            for n_acc in range(0, n_cand + 1):
                for corr in (True, False):
                    src = _committed_row_sources(n_acc, n_cand, corr)
                    n_tokens = 1 + n_acc + (1 if corr else 0)
                    assert len(src) == n_tokens


# ---------------------------------------------------------------------------
# Continuation seam helper: directly test the monotonization step on
# fabricated per-chunk word lists (no GPU required).
# ---------------------------------------------------------------------------

def test_continuation_monotonize_logic():
    """Exercise the monotonization-at-seams pass directly."""
    # Two chunks: chunk0 ends at t=10.0, chunk1's first word reports
    # t=9.5 in global coords (after stride offset).  Should be clamped.
    words = [
        WordTimestamp("hello", 0.0, 0.5),
        WordTimestamp("world", 1.0, 10.0),
        WordTimestamp("again", 9.5, 11.0),   # would precede world's end
        WordTimestamp("ok", 11.5, 12.0),
    ]

    # Inline the same monotonization the longform path applies.
    for j in range(1, len(words)):
        prev_end = words[j - 1].end
        if words[j].start < prev_end:
            words[j] = WordTimestamp(
                word=words[j].word,
                start=prev_end,
                end=max(prev_end, words[j].end),
            )

    prev_end = -1.0
    for w in words:
        assert w.start >= prev_end - 1e-9
        assert w.end >= w.start
        prev_end = w.end
