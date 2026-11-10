"""Tests for the forced-aligner timing backend (:mod:`crisperwhisper.alignment`).

The pure-Python tests drive :class:`ForcedAligner` with a stub nyra
``Aligner`` (no model, no GPU) and check the word mapping, the attention
fallbacks and that the result format is unchanged.  The ``gpu``-marked test
runs the real aligner model (downloaded from the Hugging Face Hub) on a
CrisperWhisper transcription.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from crisperwhisper import CrisperWhisperModel, ForcedAligner
from crisperwhisper.alignment import apply_timing_method
from crisperwhisper.result import (
    PhoneTimestamp,
    TokenProbability,
    TranscriptionResult,
    WordTimestamp,
)

AUDIO = np.zeros(16_000 * 5, dtype=np.float32)


def _wt(word, start, end, **kw):
    return WordTimestamp(word=word, start=start, end=end, **kw)


def _fa_word(word, start, end, type="w", confidence=0.9):
    from nyra_align import Phone, Word

    return Word(word=word, start=start, end=end, type=type,
                confidence=confidence, phones=[Phone("AH", start, end)])


class _StubNyraAligner:
    """Mimics ``nyra_align.Aligner.align``: one word per whitespace token,
    ``word`` = the raw token, tokens in ``unpronounceable`` dropped."""

    def __init__(self, unpronounceable=(), fail=False):
        self.unpronounceable = set(unpronounceable)
        self.fail = fail  # True, or a set of texts that fail
        self.texts = []
        self.audio_lens = []

    def align(self, audio, text):
        self.texts.append(text)
        self.audio_lens.append(len(audio) / 16_000)
        if self.fail is True or (self.fail and text in self.fail):
            raise RuntimeError("alignment failed at all beams")
        words = []
        for i, tok in enumerate(text.split()):
            if tok in self.unpronounceable:
                continue
            kind = "f" if tok.upper() in ("[UM]", "[UH]") else "w"
            words.append(_fa_word(tok, 10.0 + i, 10.5 + i, type=kind))
        return SimpleNamespace(words=words, skipped=[], duration=len(audio) / 16_000)


def _aligner(stub, language="en", **kw):
    fa = ForcedAligner(**kw)
    fa._aligners[language] = stub
    return fa


def _result(words, language="en"):
    return TranscriptionResult(
        text=" ".join(w.word for w in words), language=language,
        mode="verbatim", duration=5.0, processing_time=0.1, words=words,
    )


# ---------------------------------------------------------------------------
# Word mapping.
# ---------------------------------------------------------------------------

def test_refine_replaces_timing_and_adds_fields():
    stub = _StubNyraAligner()
    words = [_wt(" so", 0.0, 0.2), _wt(" [UM]", 0.3, 0.5), _wt(" yes", 0.6, 0.9)]
    out = _aligner(stub).refine(words, AUDIO, language="en")

    assert stub.texts == ["so [UM] yes"]
    assert [w.word for w in out] == [" so", " [UM]", " yes"]  # surface kept
    assert [(w.start, w.end) for w in out] == [(10.0, 10.5), (11.0, 11.5), (12.0, 12.5)]
    assert [w.kind for w in out] == ["word", "filler", "word"]
    assert all(w.timing_confidence == 0.9 for w in out)
    assert out[0].phones == [PhoneTimestamp("AH", 10.0, 10.5)]


def test_refine_keeps_asr_confidence_fields():
    toks = [TokenProbability(" hi", 7, 0.8)]
    words = [_wt(" hi", 0.0, 0.2, probability=0.8, tokens=toks)]
    out = _aligner(_StubNyraAligner()).refine(words, AUDIO, language="en")
    assert out[0].probability == 0.8 and out[0].tokens == toks
    assert out[0].start == 10.0


def test_unpronounceable_word_keeps_attention_timing():
    stub = _StubNyraAligner(unpronounceable={"&"})
    words = [_wt("a", 0.0, 0.2), _wt("&", 10.6, 10.8), _wt("b", 0.6, 0.9)]
    out = _aligner(stub).refine(words, AUDIO, language="en")
    assert (out[0].start, out[2].start) == (10.0, 12.0)
    assert (out[1].start, out[1].end) == (10.6, 10.8)  # attention, fits the gap
    assert out[1].timing_confidence is None and out[1].kind is None
    assert out[1].phones is None


def test_unaligned_word_is_clamped_between_aligned_neighbours():
    stub = _StubNyraAligner(unpronounceable={"&"})
    # attention put "&" before the previous word's forced-aligned end
    words = [_wt("a", 0.0, 0.2), _wt("&", 0.1, 0.3), _wt("b", 0.6, 0.9)]
    out = _aligner(stub).refine(words, AUDIO, language="en")
    starts = [w.start for w in out]
    ends = [w.end for w in out]
    assert starts == sorted(starts)
    assert out[1].start >= out[0].end and out[1].end <= out[2].start
    assert (out[0].start, out[0].end, out[2].start) == (10.0, 10.5, 12.0)


def test_repeated_words_with_drops_map_in_order():
    stub = _StubNyraAligner(unpronounceable={"--"})
    words = [_wt("no", 0, 0.1), _wt("--", 0.1, 0.2), _wt("no", 0.2, 0.3),
             _wt("no", 0.3, 0.4)]
    out = _aligner(stub).refine(words, AUDIO, language="en")
    # "--" keeps its attention timing, clamped to the 10.5 s end of "no"
    assert [w.start for w in out] == [10.0, 10.5, 12.0, 13.0]
    assert [w.timing_confidence for w in out] == [0.9, None, 0.9, 0.9]


def test_word_with_inner_whitespace_spans_its_pieces():
    words = [_wt(" New York", 0.0, 0.5), _wt(" city", 0.6, 0.9)]
    out = _aligner(_StubNyraAligner()).refine(words, AUDIO, language="en")
    assert (out[0].start, out[0].end) == (10.0, 11.5)
    assert len(out[0].phones) == 2
    assert out[1].start == 12.0


# ---------------------------------------------------------------------------
# Segmented alignment (attention pauses split the transcript).
# ---------------------------------------------------------------------------

LONG_AUDIO = np.zeros(16_000 * 60, dtype=np.float32)


def test_segments_split_at_long_attention_pauses():
    fa = ForcedAligner(segment_gap=2.0, segment_pad=0.5)
    words = [_wt("a", 32.0, 32.5), _wt("b", 32.6, 33.0),   # one segment
             _wt("c", 40.0, 40.4), _wt("d", 58.0, 59.8)]
    assert fa._segments(words, 60.0) == [
        (0, 2, 31.5, 33.5),
        (2, 3, 39.5, 40.9),
        (3, 4, 57.5, 60.0),   # clamped to the audio end
    ]


def test_segment_pad_never_crosses_the_pause_midpoint():
    fa = ForcedAligner(segment_gap=2.0, segment_pad=5.0)
    words = [_wt("a", 1.0, 2.0), _wt("b", 6.0, 7.0)]
    assert fa._segments(words, 10.0) == [(0, 1, 0.0, 4.0), (1, 2, 4.0, 10.0)]


def test_default_is_one_pass_over_the_file():
    fa = ForcedAligner()
    words = [_wt("a", 1.0, 2.0), _wt("b", 30.0, 31.0)]
    assert fa._segments(words, 60.0) == [(0, 2, 0.5, 31.5)]


def test_segmented_refine_crops_audio_and_shifts_times():
    stub = _StubNyraAligner()
    # leading 30 s of untranscribed audio before "ok"
    words = [_wt("ok", 32.0, 32.3), _wt("so", 32.4, 32.6), _wt("yes", 45.0, 45.4)]
    out = _aligner(stub, segment_gap=2.0, segment_pad=0.5).refine(
        words, LONG_AUDIO, language="en")
    assert stub.texts == ["ok so", "yes"]
    # crops 31.5-33.1 s and 44.5-45.9 s
    assert stub.audio_lens == pytest.approx([1.6, 1.4])
    # stub times are 10 + token index, relative to each crop's start
    assert [(w.start, w.end) for w in out] == [
        (41.5, 42.0), (42.5, 43.0), (54.5, 55.0)]
    assert out[0].phones == [PhoneTimestamp("AH", 41.5, 42.0)]


def test_failed_segment_keeps_attention_timing():
    stub = _StubNyraAligner(fail={"yes"})
    words = [_wt("ok", 1.0, 1.3), _wt("yes", 20.0, 20.4)]
    with pytest.warns(UserWarning, match="1 of 2 transcript segments"):
        out = _aligner(stub, segment_gap=2.0).refine(words, LONG_AUDIO, language="en")
    assert out[0].timing_confidence == 0.9
    assert (out[1].start, out[1].end, out[1].timing_confidence) == (20.0, 20.4, None)


def test_all_segments_failing_raises():
    stub = _StubNyraAligner(fail=True)
    words = [_wt("ok", 1.0, 1.3), _wt("yes", 20.0, 20.4)]
    with pytest.raises(RuntimeError):
        _aligner(stub, segment_gap=2.0).refine(words, LONG_AUDIO, language="en")


# ---------------------------------------------------------------------------
# apply_timing_method: fallbacks and format.
# ---------------------------------------------------------------------------

def test_attention_method_leaves_words_untouched():
    words = [_wt("hi", 0.0, 0.2)]
    res = apply_timing_method(_result(list(words)), None, "attention",
                              lambda: pytest.fail("aligner must not be built"))
    assert res.words == words
    assert res.timing_method == "attention"
    w = res.words[0]
    assert (w.timing_confidence, w.kind, w.phones) == (None, None, None)


def test_no_words_means_no_timing_method():
    res = _result([])
    res.words = None
    apply_timing_method(res, AUDIO, "forced_aligner",
                        lambda: _aligner(_StubNyraAligner()))
    assert res.timing_method is None


def test_unsupported_language_warns_even_without_words():
    res = _result([], language="de")
    res.words = None
    with pytest.warns(UserWarning, match="supports only"):
        apply_timing_method(res, AUDIO, "forced_aligner",
                            lambda: _aligner(_StubNyraAligner()))


def test_forced_aligner_method_sets_timing_method():
    res = apply_timing_method(_result([_wt("hi", 0.0, 0.2)]), AUDIO,
                              "forced_aligner",
                              lambda: _aligner(_StubNyraAligner()))
    assert res.timing_method == "forced_aligner"
    assert res.words[0].start == 10.0


def test_unsupported_language_warns_and_keeps_attention():
    words = [_wt("hallo", 0.0, 0.2)]
    stub = _StubNyraAligner()
    with pytest.warns(UserWarning, match="supports only"):
        res = apply_timing_method(_result(list(words), language="de"), AUDIO,
                                  "forced_aligner", lambda: _aligner(stub))
    assert res.words == words and res.timing_method == "attention"
    assert stub.texts == []


def test_alignment_failure_warns_and_keeps_attention():
    words = [_wt("hi", 0.0, 0.2)]
    with pytest.warns(UserWarning, match="Forced alignment failed"):
        res = apply_timing_method(_result(list(words)), AUDIO, "forced_aligner",
                                  lambda: _aligner(_StubNyraAligner(fail=True)))
    assert res.words == words and res.timing_method == "attention"


def test_language_codes_are_normalized():
    fa = ForcedAligner()
    assert fa.supports("en") and fa.supports("EN-us") and fa.supports("en_GB")
    assert not fa.supports("de")


# ---------------------------------------------------------------------------
# Standalone ForcedAligner.align.
# ---------------------------------------------------------------------------

def test_standalone_align():
    stub = _StubNyraAligner(unpronounceable={"..."})
    res = _aligner(stub).align(AUDIO, "so [um] ... yes")
    assert res.mode == "forced_align" and res.timing_method == "forced_aligner"
    assert [w.word for w in res.words] == ["so", "[um]", "yes"]
    assert [w.kind for w in res.words] == ["word", "filler", "word"]
    assert res.duration == 5.0


def test_standalone_align_rejects_unsupported_language():
    with pytest.raises(ValueError, match="does not support"):
        ForcedAligner().align(AUDIO, "hallo", language="de")


# ---------------------------------------------------------------------------
# CrisperWhisperModel wiring (engine stubbed out).
# ---------------------------------------------------------------------------

def _model(stub):
    m = CrisperWhisperModel.__new__(CrisperWhisperModel)
    m._model_version = 2
    m._backend = "ct2"
    m._model_path = "stub"
    m._forced_aligner = _aligner(stub)
    m.calls = []

    def fake_v2(audio, **kw):
        m.calls.append((audio, kw))
        words = [_wt(" hi", 0.0, 0.2), _wt(" there", 0.3, 0.6)]
        return TranscriptionResult(
            text="hi there", language=kw["language"], mode=kw["mode"],
            duration=5.0, processing_time=0.1,
            words=words if kw["word_timestamps"] else None,
        )

    m._transcribe_v2 = fake_v2
    return m


def test_transcribe_forced_aligner():
    stub = _StubNyraAligner()
    m = _model(stub)
    res = m.transcribe(AUDIO, timing_method="forced_aligner")
    audio_arg, kw = m.calls[0]
    assert kw["word_timestamps"] is True  # implied
    assert isinstance(audio_arg, np.ndarray) and kw["sr"] is None
    assert res.timing_method == "forced_aligner"
    assert [w.start for w in res.words] == [10.0, 11.0]
    assert stub.texts == ["hi there"]


def test_transcribe_default_is_attention():
    m = _model(_StubNyraAligner())
    res = m.transcribe(AUDIO, word_timestamps=True)
    assert res.timing_method == "attention"
    assert [w.start for w in res.words] == [0.0, 0.3]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert m.transcribe(AUDIO).timing_method is None


def test_transcribe_rejects_unknown_timing_method():
    with pytest.raises(ValueError, match="timing_method"):
        _model(_StubNyraAligner()).transcribe(AUDIO, timing_method="viterbi")


def test_transcribe_forced_aligner_non_english_warns():
    m = _model(_StubNyraAligner())
    with pytest.warns(UserWarning, match="supports only"):
        res = m.transcribe(AUDIO, language="de", timing_method="forced_aligner")
    assert res.timing_method == "attention"
    assert [w.start for w in res.words] == [0.0, 0.3]


def test_forced_align_delegates_to_aligner():
    stub = _StubNyraAligner()
    m = _model(stub)
    res = m.forced_align(AUDIO, "hello world", timing_method="forced_aligner")
    assert m.calls == []  # no transcription pass
    assert res.timing_method == "forced_aligner"
    assert [w.word for w in res.words] == ["hello", "world"]


# ---------------------------------------------------------------------------
# Real aligner end to end (downloads the aligner + needs a CT2 model).
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_e2e_forced_aligner_matches_direct_alignment(model, en_audio):
    import nyra_align

    from crisperwhisper.audio import load_audio

    res = model.transcribe(en_audio, timing_method="forced_aligner")
    assert res.timing_method == "forced_aligner"
    assert res.words and all(w.timing_confidence is not None for w in res.words)
    assert all(w.kind and w.phones for w in res.words)

    text = " ".join(w.word.strip() for w in res.words)
    direct = nyra_align.Aligner().align(load_audio(en_audio), text)
    assert [(w.start, w.end) for w in direct.words] == [
        (w.start, w.end) for w in res.words
    ]
