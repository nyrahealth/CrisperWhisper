"""Find decoded words that were never spoken.

Loop repair can make the decoder re-emit text it has already transcribed.
The word aligner cannot find such text in the audio: it either leaves the
words unplaced or squashes them onto a single instant. Decoded words that
have no room in the audio, or that loop over what was just said without
being placed, are not speech and are dropped from the transcript.

Measured on 16 production recordings that went through loop repair, scored
against an independent reference transcript: this removes 397 invented
tokens and 12 real ones (no real cut-offs or sound events), WER 12.9% -> 10.8%.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

#: Least audio per word for a run of unplaced words to be kept. Fast speech is
#: ~5 words/s (200 ms per word); 50 ms is four times faster than that.
UNPLACED_MIN_SECONDS = 0.05

#: A timed word shorter than this was stamped onto an instant rather than
#: placed, and counts as unplaced when looking for runs without room.
SQUASHED_MAX_SECONDS = 0.02

#: An unplaced run of at least ``LOOP_MIN_UNPLACED`` words that repeats a
#: ``LOOP_NGRAM``-word sequence already said in the run, or in the
#: ``LOOP_CONTEXT`` words before it, is a decoder loop -- dropped even where
#: the audio would have room for it.
LOOP_MIN_UNPLACED = 8
LOOP_NGRAM = 4
LOOP_CONTEXT = 20


class _Timed(Protocol):
    start: float | None
    end: float | None


def _loop_key(word: str) -> str:
    """Comparable form of a word for loop detection; tags compare as nothing."""
    if word.startswith("["):
        return ""
    return re.sub(r"[^\w']", "", word.lower())


def _repeats_itself(words: Sequence[str], start: int, end: int) -> bool:
    """Whether a ``LOOP_NGRAM`` starting inside ``words[start:end]`` was said
    before, earlier in the run or in the ``LOOP_CONTEXT`` words ahead of it."""
    before = words[max(0, start - LOOP_CONTEXT):start]
    context = [key for key in map(_loop_key, before) if key]
    keys = context + [key for key in map(_loop_key, words[start:end]) if key]
    seen: set[tuple[str, ...]] = set()
    for position in range(len(keys) - LOOP_NGRAM + 1):
        gram = tuple(keys[position:position + LOOP_NGRAM])
        if position >= len(context) and gram in seen:
            return True
        seen.add(gram)
    return False


def unspoken_runs(
    words: Sequence[str],
    timings: Sequence[_Timed | None],
    duration: float,
) -> set[int]:
    """Positions of word runs that were decoded but never spoken.

    ``timings`` is 1-to-1 with ``words``: the word's timing, or ``None`` (or
    a ``None`` start) where the aligner could not place it. Timed words
    shorter than ``SQUASHED_MAX_SECONDS`` count as unplaced too. A run's room
    is the gap between the end of the placed word before it and the start of
    the placed word after it (the recording's ends at the edges). A run is
    dropped when

    * it has no room: under ``UNPLACED_MIN_SECONDS`` per word, with at least
      two words unplaced or three squashed, or
    * it is a loop: ``LOOP_MIN_UNPLACED`` or more unplaced words repeating
      what was just said.
    """
    if len(words) != len(timings):
        raise ValueError("words and timings must be 1-to-1")
    missing = [t is None or t.start is None or t.end is None for t in timings]
    unplaced = [
        gone or t.end - t.start < SQUASHED_MAX_SECONDS  # type: ignore[union-attr,operator]
        for gone, t in zip(missing, timings)
    ]
    dropped: set[int] = set()
    index = 0
    while index < len(words):
        if not unplaced[index]:
            index += 1
            continue
        run_end = index
        while run_end < len(words) and unplaced[run_end]:
            run_end += 1
        length = run_end - index
        n_missing = sum(missing[index:run_end])
        left = timings[index - 1].end if index else 0.0  # type: ignore[union-attr]
        right = timings[run_end].start if run_end < len(words) else duration  # type: ignore[union-attr]
        no_room = (
            (n_missing >= 2 or length >= 3)
            and max(right - left, 0.0) / length < UNPLACED_MIN_SECONDS
        )
        loop = n_missing >= LOOP_MIN_UNPLACED and _repeats_itself(words, index, run_end)
        if no_room or loop:
            dropped.update(range(index, run_end))
        index = run_end
    return dropped
