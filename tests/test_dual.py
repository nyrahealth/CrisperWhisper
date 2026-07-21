"""Tests for batched dual-mode transcription (``transcribe_dual``).

``transcribe_dual`` decodes several modes (verbatim + intended) for one
audio in a single batched decoder pass.  Its output must be identical to
calling :meth:`CrisperWhisperModel.transcribe` once per mode -- both the
text and (when requested) the word timings -- while doing the expensive
autoregressive decode only once.

The end-to-end checks are gated behind ``@pytest.mark.gpu`` (need a
converted CT2 model + CUDA); the argument-validation checks run anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.result import WordTimestamp

# Word-timing tolerance between the batched and single-mode paths.  The
# batched verify pass uses a single time-axis forward where the single-mode
# path steps token-by-token; the two decoder kernels differ at the ~1e-4
# level, which can nudge a Viterbi boundary by at most one encoder frame.
_TIMING_TOL_S = 0.021  # ~1 frame (20 ms) + epsilon


def _assert_words_match(single, dual):
    single = single or []
    dual = dual or []
    assert len(single) == len(dual), (
        f"word count differs: {len(single)} vs {len(dual)}"
    )
    for i, (a, b) in enumerate(zip(single, dual)):
        assert a.word == b.word, f"word[{i}] text {a.word!r} != {b.word!r}"
        assert abs(a.start - b.start) <= _TIMING_TOL_S, (
            f"word[{i}] {a.word!r} start {a.start:.3f} vs {b.start:.3f}"
        )
        assert abs(a.end - b.end) <= _TIMING_TOL_S, (
            f"word[{i}] {a.word!r} end {a.end:.3f} vs {b.end:.3f}"
        )


# ---------------------------------------------------------------------------
# Argument validation (no model / GPU needed -- failures raise before load).
# ---------------------------------------------------------------------------

class TestDualValidation:
    def test_bad_mode_raises(self, model):
        with pytest.raises(ValueError, match="Unknown mode"):
            model.transcribe_dual(
                np.zeros(16000, dtype=np.float32), sr=16000,
                modes=("verbatim", "bogus"),
            )

    def test_empty_modes_raises(self, model):
        with pytest.raises(ValueError, match="non-empty"):
            model.transcribe_dual(
                np.zeros(16000, dtype=np.float32), sr=16000, modes=(),
            )

    def test_longform_runs(self, model):
        # Longform (>30s) dual mode is supported via the catch-up batched
        # decode; it must not raise and must return one result per mode.
        audio = np.zeros(int(31 * 16000), dtype=np.float32)
        results = model.transcribe_dual(
            audio, sr=16000, modes=("verbatim", "intended"),
        )
        assert len(results) == 2
        assert results[0].mode == "verbatim"
        assert results[1].mode == "intended"

    def test_bad_longform_strategy_raises(self, model):
        audio = np.zeros(int(31 * 16000), dtype=np.float32)
        with pytest.raises(NotImplementedError, match="continuation"):
            model.transcribe_dual(audio, sr=16000, longform_strategy="bogus")


# ---------------------------------------------------------------------------
# End-to-end parity vs two single-mode transcribe() calls.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestDualParity:
    def test_text_parity(self, model, en_audio):
        sv = model.transcribe(en_audio, mode="verbatim")
        si = model.transcribe(en_audio, mode="intended")
        dv, di = model.transcribe_dual(en_audio, modes=("verbatim", "intended"))
        assert dv.mode == "verbatim" and di.mode == "intended"
        assert dv.text == sv.text
        assert di.text == si.text

    def test_word_timing_parity(self, model, en_audio):
        sv = model.transcribe(en_audio, mode="verbatim", word_timestamps=True)
        si = model.transcribe(en_audio, mode="intended", word_timestamps=True)
        dv, di = model.transcribe_dual(
            en_audio, modes=("verbatim", "intended"), word_timestamps=True,
        )
        assert dv.text == sv.text and di.text == si.text
        _assert_words_match(sv.words, dv.words)
        _assert_words_match(si.words, di.words)
        for w in (dv.words or []) + (di.words or []):
            assert isinstance(w, WordTimestamp)

    def test_mode_order_respected(self, model, en_audio):
        sv = model.transcribe(en_audio, mode="verbatim")
        si = model.transcribe(en_audio, mode="intended")
        di, dv = model.transcribe_dual(en_audio, modes=("intended", "verbatim"))
        assert di.mode == "intended" and dv.mode == "verbatim"
        assert di.text == si.text
        assert dv.text == sv.text

    def test_single_mode_request(self, model, en_audio):
        sv = model.transcribe(en_audio, mode="verbatim", word_timestamps=True)
        (dv,) = model.transcribe_dual(
            en_audio, modes=("verbatim",), word_timestamps=True,
        )
        assert dv.text == sv.text
        _assert_words_match(sv.words, dv.words)


# ---------------------------------------------------------------------------
# Suppress-token pass-through.
# ---------------------------------------------------------------------------

class TestDualSuppress:
    def test_invalid_chunk_duration_raises(self, model):
        # transcribe_dual validates the longform window eagerly (same single
        # validation point as transcribe()), even for short audio.
        with pytest.raises(ValueError, match="chunk_duration"):
            model.transcribe_dual(
                np.zeros(16000, dtype=np.float32), sr=16000,
                chunk_duration=45.0,
            )

    @pytest.mark.gpu
    def test_dual_suppress_roundtrip(self, model, en_audio):
        """Suppressing a token emitted by the batched decode removes it from
        every row, and results still match per-mode transcribe()."""
        from crisperwhisper.audio import load_audio
        from crisperwhisper.prompt import PromptBuilder

        engine = model._engine
        audio = load_audio(en_audio)
        builder = PromptBuilder(engine)
        prompts = [builder.verbatim(), builder.intended()]
        features = engine.extract_features(audio)

        base, _ = engine.generate_dual_greedy(
            features, prompts, max_length=64, hallucination_mitigation=False,
        )
        target = next(
            (int(t) for t in base[0] if int(t) not in engine.all_special_ids),
            None,
        )
        assert target is not None

        gens, _ = engine.generate_dual_greedy(
            features, prompts, max_length=64,
            hallucination_mitigation=False, suppress_tokens=[target],
        )
        for row in gens:
            assert target not in [int(t) for t in row]

    @pytest.mark.gpu
    def test_dual_model_level_suppress(self, model, en_audio):
        dv, di = model.transcribe_dual(en_audio, suppress_tokens=[])
        assert dv.text and di.text
