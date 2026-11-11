"""Dropping decoded words that were never spoken (crisperwhisper.unspoken)."""

from crisperwhisper.longform.base import LongformConfig
from crisperwhisper.longform.continuation import _finalize_timeline
from crisperwhisper.result import WordTimestamp
from crisperwhisper.unspoken import unspoken_runs


def ts(word, start, end):
    return WordTimestamp(word=word, start=start, end=end)


def build(*items):
    """(word, start, end) or bare word for an unplaced one -> words, timings."""
    words, timings = [], []
    for item in items:
        if isinstance(item, str):
            words.append(item)
            timings.append(None)
        else:
            words.append(item[0])
            timings.append(ts(*item))
    return words, timings


def kept(words, timings, duration):
    dropped = unspoken_runs(words, timings, duration)
    return [w for i, w in enumerate(words) if i not in dropped]


def test_an_unplaced_repeat_with_no_room_is_dropped():
    """Production req_f8116d3c...: loop repair re-emitted a passage it had
    already transcribed and none of it could be placed."""
    words, timings = build(
        ("for", 0.0, 0.2), ("now", 0.2, 0.4), ("we", 0.4, 0.6),
        *"for now we are going to use just voice".split(),
        ("but", 0.6, 0.8), ("then", 0.8, 1.0),
    )
    assert kept(words, timings, 1.0) == ["for", "now", "we", "but", "then"]


def test_unplaced_words_with_room_are_kept():
    words, timings = build(("one", 0.0, 0.2), "two", "three", ("four", 0.8, 1.0))
    assert kept(words, timings, 1.0) == words


def test_a_single_unplaced_word_is_kept_even_without_room():
    words, timings = build(("one", 0.0, 0.5), "two", ("three", 0.5, 1.0))
    assert kept(words, timings, 1.0) == words


def test_a_loop_squashed_onto_one_instant_is_dropped():
    """Production req_3936e98a...: the aligner stamped a loop onto 420.0 s."""
    words, timings = build(
        ("how", 0.0, 0.1), ("things", 0.1, 0.2), ("are", 0.2, 0.3),
        ("going", 0.3, 0.3), ("with", 0.3, 0.3), ("the", 0.3, 0.3),
        ("customer", 0.3, 0.3), ("and", 0.3, 0.4), ("what", 0.4, 1.0),
    )
    assert kept(words, timings, 1.0) == ["how", "things", "are", "and", "what"]


def test_one_or_two_very_short_words_in_fast_speech_are_kept():
    words, timings = build(
        ("it", 0.0, 0.2), ("is", 0.2, 0.4), ("a", 0.4, 0.41),
        ("b", 0.41, 0.42), ("test", 0.42, 1.0),
    )
    assert kept(words, timings, 1.0) == words


def test_an_unplaced_loop_is_dropped_even_with_room_for_it():
    """Production req_9c69dc65...: three unplaced repeats of what had just
    been said, over a 2.6 s gap that would have held them."""
    loop = "the platform and [UM] to get your thoughts on".split() * 3
    words, timings = build(
        ("to", 0.0, 0.1), ("get", 0.1, 0.2), ("your", 0.2, 0.3),
        ("thoughts", 0.3, 0.4), ("on", 0.4, 0.5),
        *loop,
        ("the", 3.1, 3.2), ("area", 3.2, 4.0),
    )
    assert kept(words, timings, 4.0) == [
        "to", "get", "your", "thoughts", "on", "the", "area",
    ]


def test_a_long_unplaced_run_that_does_not_repeat_is_kept():
    words, timings = build(
        ("we", 0.0, 0.2), ("start", 0.2, 0.4),
        *"one two three four five six seven eight nine ten".split(),
        ("and", 3.4, 3.6), ("end", 3.6, 4.0),
    )
    assert kept(words, timings, 4.0) == words


def test_repeated_placed_speech_is_never_touched():
    """Real repetitions the aligner placed are speech, however loopy."""
    phrase = ["you", "know", "what", "I", "mean"] * 3
    words, timings = build(*[(w, 0.3 * i, 0.3 * i + 0.25) for i, w in enumerate(phrase)])
    assert kept(words, timings, 5.0) == words


def test_unplaced_words_at_the_edges_use_the_recording_bounds():
    words, timings = build("lost", "words", ("hello", 0.0, 0.5), ("end", 0.5, 1.0))
    assert kept(words, timings, 1.0) == ["hello", "end"]
    words, timings = build(("hello", 0.0, 0.5), "late", "words")
    assert kept(words, timings, 1.0) == words


def test_finalize_drops_from_text_and_timings_after_monotonizing():
    words, timeline = build(
        ("one", 0.0, 0.5), "x", "y", "z", ("two", 0.48, 0.9), "kept",
    )
    out_words, out_ts = _finalize_timeline(
        words, timeline, 2.0, drop_unspoken=True, mode="verbatim",
    )
    assert out_words == ["one", "two", "kept"]
    assert [(t.word, t.start, t.end) for t in out_ts] == [
        ("one", 0.0, 0.5), ("two", 0.5, 0.9),
    ]


def test_finalize_can_be_switched_off():
    words, timeline = build(("one", 0.0, 0.5), "x", "y", ("two", 0.5, 1.0))
    out_words, out_ts = _finalize_timeline(
        words, timeline, 1.0, drop_unspoken=False, mode="verbatim",
    )
    assert out_words == words and len(out_ts) == 2


def test_dropping_is_on_by_default():
    assert LongformConfig().drop_unspoken is True
