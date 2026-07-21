"""End-to-end GPU tests for the CrisperWhisper package.

Exercises the public ``CrisperWhisperModel`` API (transcription modes,
hotwords, verbatimize, longform strategies, engine internals, RTF) against
a real converted model on CUDA.

Skipped unless ``CW2_MODEL_PATH`` is set (see ``conftest.py``).  Audio comes
from the checked-in ``samples/`` directory, so no external data is needed;
longform audio is synthesised by tiling the sample past 30 s.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def long_audio(en_audio):
    """A >30 s mono 16 kHz array (tiled from the English sample) to force
    the longform path."""
    from crisperwhisper.audio import load_audio

    base = load_audio(en_audio)  # 16 kHz float32
    reps = int(np.ceil(31.0 * 16000 / len(base)))
    return np.tile(base, reps)


class TestShortAudio:
    def test_verbatim_mode(self, model, en_audio):
        result = model.transcribe(en_audio, mode="verbatim", language="en")
        assert result.text, "empty transcription"
        assert result.duration > 0
        assert result.processing_time > 0
        assert result.mode == "verbatim"

    def test_intended_mode(self, model, en_audio):
        result = model.transcribe(en_audio, mode="intended", language="en")
        assert result.text, "empty transcription"
        assert result.mode == "intended"

    def test_with_hotwords(self, model, en_audio):
        result = model.transcribe(
            en_audio, mode="verbatim", language="en",
            hotwords=["microphone", "adjustments"],
        )
        assert result.text, "empty transcription"

    def test_no_hallucination_mitigation(self, model, en_audio):
        result = model.transcribe(
            en_audio, mode="verbatim", language="en",
            hallucination_mitigation=False,
        )
        assert result.text, "empty transcription"

    def test_numpy_input(self, model, en_audio):
        from crisperwhisper.audio import load_audio

        audio_arr = load_audio(en_audio)
        result = model.transcribe(
            audio_arr, mode="verbatim", language="en", sr=16000,
        )
        assert result.text, "empty transcription"


class TestLongformAudio:
    def test_continuation_strategy(self, model, long_audio):
        result = model.transcribe(
            long_audio, sr=16000, mode="verbatim", language="en",
            longform_strategy="continuation",
        )
        assert result.text, "empty transcription"
        assert result.duration > 30
        assert result.chunks is not None
        assert len(result.chunks) > 1

    def test_chunked_lcs_strategy(self, model, long_audio):
        result = model.transcribe(
            long_audio, sr=16000, mode="verbatim", language="en",
            longform_strategy="chunked_lcs",
        )
        assert result.text, "empty transcription"
        assert result.chunks is not None

    def test_token_lcs_strategy(self, model, long_audio):
        result = model.transcribe(
            long_audio, sr=16000, mode="verbatim", language="en",
            longform_strategy="token_lcs",
        )
        assert result.text, "empty transcription"
        assert result.chunks is not None


class TestVerbatimize:
    def test_verbatimize(self, model, en_audio):
        result = model.verbatimize(
            en_audio, "the microphone adjustments we should check", language="en",
        )
        assert result.text, "empty verbatimize output"
        assert result.mode == "verbatimize"

    def test_verbatimize_word_timestamps(self, model, en_audio):
        """Verbatimize captures word timings inline from its own decode."""
        result = model.verbatimize(
            en_audio, "the microphone adjustments we should check",
            language="en", word_timestamps=True,
        )
        assert result.text, "empty verbatimize output"
        assert result.words, "word_timestamps=True produced no words"
        for i, w in enumerate(result.words):
            assert w.start is not None and w.end is not None
            assert w.end >= w.start
            if i:
                assert w.start >= result.words[i - 1].start - 1e-6


class TestEngineDirectly:
    def test_feature_extraction(self, model, en_audio):
        from crisperwhisper.audio import load_audio

        audio = load_audio(en_audio)
        features = model._engine.extract_features(audio)
        assert len(features.shape) == 3  # (1, n_mels, n_frames)

    def test_tokenizer_special_tokens(self, model):
        engine = model._engine
        assert engine.eot_id is not None
        assert engine.sot_id is not None
        assert engine.no_timestamps_id is not None

        vocab = engine.tokenizer.get_vocab()
        for tok in ("[verbatim_1]", "<vtx>", "<htx>", "<ctx>"):
            assert tok in vocab, f"{tok} not in vocabulary"

    def test_prompt_builder(self, model):
        from crisperwhisper.prompt import PromptBuilder

        pb = PromptBuilder(model._engine, language="en")
        verbatim_tokens = pb.verbatim()
        assert len(pb.verbatim(hotwords=["test"])) > len(verbatim_tokens)
        assert len(pb.verbatim(context="previous words")) > len(verbatim_tokens)
        assert len(pb.verbatimize("hello world")) > len(verbatim_tokens)


class TestPerformance:
    def test_rtf(self, model, en_audio):
        """Real-time factor should be well below 1.0 on GPU."""
        result = model.transcribe(en_audio, mode="verbatim", language="en")
        rtf = (
            result.processing_time / result.duration
            if result.duration > 0 else float("inf")
        )
        assert rtf < 1.0, f"RTF {rtf:.2f} >= 1.0 (slower than real-time)"
