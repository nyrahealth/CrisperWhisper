"""Tests for forced alignment (:mod:`crisperwhisper.forced_align`).

Two layers:

* Pure-Python tests drive the transcribe-then-align core (word matching +
  gap interpolation) with hand-built hypothesis words -- no model, no GPU.
* Backend-marked self-consistency tests align a model's own transcript back to
  the audio and check the forced timings reproduce the transcription timings.
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.forced_align import align_to_hypothesis, default_normalize
from crisperwhisper.result import WordTimestamp


def _wt(word, start, end):
    return WordTimestamp(word=word, start=start, end=end)


# ---------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------

def test_default_normalize():
    assert default_normalize("Warranty?") == "warranty"
    assert default_normalize("[UM]") == "um"
    assert default_normalize("that's") == "thats"
    assert default_normalize("...") == ""


# ---------------------------------------------------------------------------
# Core alignment: exact-match passthrough.
# ---------------------------------------------------------------------------

def test_exact_match_inherits_timestamps():
    hyp = [_wt("hello", 0.0, 0.5), _wt("world", 0.6, 1.0)]
    out = align_to_hypothesis("Hello world", hyp, audio_duration=1.2)
    assert [w.word for w in out] == ["Hello", "world"]  # surface preserved
    assert out[0].start == 0.0 and out[0].end == 0.5
    assert out[1].start == 0.6 and out[1].end == 1.0


def test_casing_and_punctuation_still_match():
    hyp = [_wt("the", 1.0, 1.1), _wt("warranty", 1.2, 1.9)]
    out = align_to_hypothesis("the Warranty?", hyp, audio_duration=3.0)
    assert out[1].word == "Warranty?"  # surface form kept
    assert out[1].start == 1.2 and out[1].end == 1.9  # matched anchor time


# ---------------------------------------------------------------------------
# Core alignment: interpolation of unmatched reference words.
# ---------------------------------------------------------------------------

def test_internal_gap_interpolated_between_anchors():
    # reference has an extra word "big" with no hypothesis match.
    hyp = [_wt("a", 0.0, 1.0), _wt("cat", 5.0, 6.0)]
    out = align_to_hypothesis("a big cat", hyp, audio_duration=7.0)
    assert [w.word for w in out] == ["a", "big", "cat"]
    # "a" and "cat" keep anchors; "big" lands strictly between them.
    assert out[0].end == 1.0
    assert out[2].start == 5.0
    assert 1.0 <= out[1].start <= out[1].end <= 5.0
    # monotonic
    assert out[0].end <= out[1].start and out[1].end <= out[2].start


def test_substitution_inherits_hypothesis_time_not_interpolated():
    # "ok" (ref) vs "okay" (hyp) is a 1:1 substitution -> should inherit the
    # hypothesis time directly, not be interpolated across neighbours.
    hyp = [_wt("hello", 0.0, 0.5), _wt("okay", 10.0, 10.6), _wt("bye", 11.0, 11.4)]
    out = align_to_hypothesis("hello OK bye", hyp, audio_duration=12.0)
    assert out[1].word == "OK"
    assert out[1].start == 10.0 and out[1].end == 10.6  # substitution anchor


def test_unequal_replace_distributed_over_replaced_span():
    # two ref words replace one hyp word -> spread across that word's span.
    hyp = [_wt("a", 0.0, 0.5), _wt("wantto", 4.0, 5.0), _wt("z", 8.0, 8.5)]
    out = align_to_hypothesis("a wanna to z", hyp, audio_duration=9.0)
    # "wanna" + "to" replace "wantto" -> both inside [4.0, 5.0]
    mids = [w for w in out if w.word in ("wanna", "to")]
    for w in mids:
        assert 4.0 <= w.start <= w.end <= 5.0 + 1e-6


def test_leading_and_trailing_gaps_interpolated():
    hyp = [_wt("middle", 4.0, 4.5)]
    out = align_to_hypothesis("intro middle outro", hyp, audio_duration=10.0)
    # leading word before [0, 4.0], trailing word after [4.5, 10.0]
    assert 0.0 <= out[0].start <= out[0].end <= 4.0
    assert out[1].start == 4.0 and out[1].end == 4.5
    assert 4.5 <= out[2].start <= out[2].end <= 10.0


def test_no_hypothesis_spreads_evenly():
    out = align_to_hypothesis("one two three", [], audio_duration=9.0)
    assert len(out) == 3
    starts = [w.start for w in out]
    assert starts == sorted(starts)
    assert out[0].start == 0.0
    assert out[-1].end <= 9.0 + 1e-6


def test_empty_text_returns_empty():
    assert align_to_hypothesis("", [_wt("x", 0.0, 1.0)], 1.0) == []


def test_output_is_monotonic_with_messy_hypothesis():
    # hypothesis slightly out of order / overlapping; output must stay monotonic.
    hyp = [_wt("a", 0.0, 2.0), _wt("b", 1.5, 1.6), _wt("c", 3.0, 4.0)]
    out = align_to_hypothesis("a b c", hyp, audio_duration=5.0)
    for j in range(1, len(out)):
        assert out[j].start >= out[j - 1].end


def test_word_count_always_matches_reference():
    hyp = [_wt("w%d" % i, i * 0.5, i * 0.5 + 0.4) for i in range(10)]
    ref = "alpha beta gamma delta epsilon"  # disjoint from hyp -> all interp
    out = align_to_hypothesis(ref, hyp, audio_duration=6.0)
    assert len(out) == len(ref.split())
    starts = [w.start for w in out]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# End-to-end self-consistency (transformers + CT2 backends).
# ---------------------------------------------------------------------------

def _assert_self_consistent(m, audio):
    ref = m.transcribe(audio, word_timestamps=True)
    assert ref.words, "need a transcript with word timestamps to align against"
    ref_text = " ".join(w.word for w in ref.words)

    aligned = m.forced_align(audio, ref_text)
    assert aligned.mode == "forced_align"
    assert aligned.words
    assert len(aligned.words) == len(ref.words)

    starts = [w.start for w in aligned.words]
    assert starts == sorted(starts)

    # Aligning a transcript to itself is an all-equal match -> timings identical.
    diffs = [abs(a.start - r.start) for a, r in zip(aligned.words, ref.words)]
    assert float(np.median(diffs)) < 0.05


@pytest.mark.transformers
def test_forced_align_self_consistency_transformers(tf_model, en_audio):
    _assert_self_consistent(tf_model, en_audio)


@pytest.mark.gpu
def test_forced_align_self_consistency_ct2(model, en_audio):
    _assert_self_consistent(model, en_audio)
