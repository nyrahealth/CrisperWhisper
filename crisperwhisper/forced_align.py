"""Forced alignment of a known transcript to audio (any length).

Given audio and the text that was spoken, this produces word-level timestamps
for the *provided* text.  The strategy is **transcribe-then-align**:

1.  Transcribe the audio with the normal (longform) pipeline, which yields the
    model's own hypothesis words *with* cross-attention word timestamps.  The
    transcription pipeline already handles pauses, hallucination repair and
    >30s chunking robustly, and the timestamps are computed on the model's own
    output where the cross-attention is sharp.
2.  Align the reference transcript to the hypothesis at the word level
    (:func:`difflib.SequenceMatcher`).  Exact matches inherit the hypothesis
    timestamp directly.  **Substitutions** (e.g. "OK"->"Okay", "will"->"would")
    still occupy the same audio span as the hypothesis words they replaced, so
    the reference words are anchored to that span -- not interpolated.
3.  Only reference words with *no* hypothesis counterpart at all (pure
    insertions/deletions -- fillers or cut-words the model dropped) are
    **interpolated** across the interval between their surrounding anchors,
    proportional to word length.

This is far more robust on real-world audio than teacher-forcing the text
through sliding windows: pauses and silences are represented for free by the
hypothesis timestamps, and a reference word can never drift further than the
gap between its two neighboring anchors (no catastrophic desync).

It also works on **every backend** (CT2, transformers, legacy v1), since it
relies only on ``transcribe(..., word_timestamps=True)``.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, List, Optional, Tuple

from crisperwhisper.result import WordTimestamp
from crisperwhisper.word_timing import monotonize_words

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def default_normalize(word: str) -> str:
    """Lower-case and strip to alphanumerics for word matching.

    e.g. ``"Warranty?"`` -> ``"warranty"``, ``"[UM]"`` -> ``"um"``,
    ``"that's"`` -> ``"thats"``.
    """
    return _PUNCT_RE.sub("", word.lower())


def align_to_hypothesis(
    reference_text: str,
    hypothesis_words: List[WordTimestamp],
    audio_duration: float,
    *,
    normalize: Callable[[str], str] = default_normalize,
) -> List[WordTimestamp]:
    """Project ``reference_text`` onto ``hypothesis_words`` timestamps.

    Parameters
    ----------
    reference_text
        The transcript to align.  Split on whitespace; each token's surface
        form is preserved in the output.
    hypothesis_words
        Timestamped words from transcription (the timing anchors).
    audio_duration
        Audio length in seconds (used to bound leading/trailing interpolation).
    normalize
        Word normalizer used for matching only (not for output).

    Returns
    -------
    One :class:`WordTimestamp` per whitespace-token of ``reference_text``,
    monotonic in time.
    """
    surface = reference_text.split()
    n = len(surface)
    if n == 0:
        return []

    if not hypothesis_words:
        # No timing information at all -> spread evenly across the audio.
        return _spread_even(surface, 0.0, audio_duration, normalize)

    ref_norm = [normalize(w) for w in surface]
    hyp_norm = [normalize(h.word) for h in hypothesis_words]

    times: List[Optional[Tuple[float, float]]] = [None] * n
    matcher = difflib.SequenceMatcher(a=ref_norm, b=hyp_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Exact match -> inherit the hypothesis timestamp 1:1.
            for di, dj in zip(range(i1, i2), range(j1, j2)):
                h = hypothesis_words[dj]
                times[di] = (float(h.start), float(h.end))
        elif tag == "replace":
            # A substitution still occupies the same audio span as the
            # hypothesis words it replaced (e.g. "OK"->"Okay", "will"->
            # "would", "Bitcom"->"Bcom").  Anchor the reference words to that
            # span rather than interpolating them across far-away neighbours.
            if (i2 - i1) == (j2 - j1):
                # equal counts -> 1:1 positional correspondence
                for di, dj in zip(range(i1, i2), range(j1, j2)):
                    h = hypothesis_words[dj]
                    times[di] = (float(h.start), float(h.end))
            else:
                # different counts -> distribute over the replaced span
                t0 = float(hypothesis_words[j1].start)
                t1 = float(hypothesis_words[j2 - 1].end)
                _distribute(times, ref_norm, i1, i2, t0, t1)
        # "delete" (reference words with no hypothesis) and "insert"
        # (hypothesis words with no reference) leave `times` as None here;
        # delete gaps are interpolated below, inserts have no reference word.

    _fill_gaps(times, ref_norm, audio_duration, normalize, surface)

    out = [
        WordTimestamp(
            word=surface[i],
            start=round(times[i][0], 3),  # type: ignore[index]
            end=round(times[i][1], 3),  # type: ignore[index]
        )
        for i in range(n)
    ]
    monotonize_words(out)
    return out


# ---------------------------------------------------------------------------
# Interpolation helpers.
# ---------------------------------------------------------------------------

def _distribute(
    times: List[Optional[Tuple[float, float]]],
    ref_norm: List[str],
    lo: int,
    hi: int,
    t0: float,
    t1: float,
) -> None:
    """Fill ``times[lo:hi]`` with contiguous segments over ``[t0, t1]``.

    Segment widths are proportional to (normalized) word length so longer
    words get proportionally more time.
    """
    if hi <= lo:
        return
    t1 = max(t1, t0)
    idxs = list(range(lo, hi))
    weights = [len(ref_norm[i]) + 1 for i in idxs]
    total = float(sum(weights)) or 1.0
    span = t1 - t0
    cur = t0
    for i, w in zip(idxs, weights):
        seg = span * (w / total)
        times[i] = (cur, cur + seg)
        cur += seg


def _fill_gaps(
    times: List[Optional[Tuple[float, float]]],
    ref_norm: List[str],
    audio_duration: float,
    normalize: Callable[[str], str],
    surface: List[str],
) -> None:
    n = len(times)
    anchors = [i for i in range(n) if times[i] is not None]
    if not anchors:
        spread = _spread_even(surface, 0.0, audio_duration, normalize)
        for i in range(n):
            times[i] = (spread[i].start, spread[i].end)
        return

    first = anchors[0]
    if first > 0:
        _distribute(times, ref_norm, 0, first, 0.0, times[first][0])  # type: ignore[index]

    for a, b in zip(anchors, anchors[1:]):
        if b - a > 1:
            _distribute(
                times, ref_norm, a + 1, b,
                times[a][1], times[b][0],  # type: ignore[index]
            )

    last = anchors[-1]
    if last < n - 1:
        end_bound = max(audio_duration, times[last][1])  # type: ignore[index]
        _distribute(times, ref_norm, last + 1, n, times[last][1], end_bound)  # type: ignore[index]


def _spread_even(
    surface: List[str],
    t0: float,
    t1: float,
    normalize: Callable[[str], str],
) -> List[WordTimestamp]:
    n = len(surface)
    span = max(0.0, t1 - t0)
    weights = [len(normalize(w)) + 1 for w in surface]
    total = float(sum(weights)) or 1.0
    cur = t0
    out: List[WordTimestamp] = []
    for w, wt in zip(surface, weights):
        seg = span * (wt / total)
        out.append(WordTimestamp(word=w, start=round(cur, 3), end=round(cur + seg, 3)))
        cur += seg
    return out


