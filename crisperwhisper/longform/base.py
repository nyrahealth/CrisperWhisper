"""Shared longform transcription utilities: config and chunking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class LongformConfig:
    """Tuneable parameters for longform transcription.

    ``chunk_duration`` is the audio window length in seconds (<= 30, the
    Whisper encoder window) and ``stride`` is how far the window advances
    between chunks; their difference is the overlap region re-covered by the
    next window.  For the continuation strategy, ``drop_words`` is the *cap* on
    how many trailing words may be dropped at a non-final chunk boundary: when
    ``timestamp_aware_drop`` is True a trailing word is dropped only if its
    audio starts inside the overlap region ``[stride, chunk_duration]`` (so the
    next window actually re-covers it) -- words ending before the stride
    boundary are confirmed, never lost.  When ``timestamp_aware_drop`` is False
    the legacy fixed-count drop (always the last ``drop_words`` words) is used.

    ``context_words`` should be large enough that the continuation context text
    spans the overlap region (``chunk_duration - stride``): at inference the next
    window re-presents the overlap audio, and the model continues cleanly only
    when its context covers what it re-hears.  At ~2.8 words/s a 4 s overlap is
    ~11 words, so the default is 12 (benchmarks: raising 8 -> 12 cut meanwhile
    boundary-region WER roughly in half with no change elsewhere).
    """

    chunk_duration: float = 30.0
    stride: float = 26.0
    context_words: int = 12
    drop_words: int = 2
    max_new_tokens: int = 256
    timestamp_aware_drop: bool = True
    temperature_fallback: bool = True


def make_chunks(audio: np.ndarray, config: LongformConfig) -> list[np.ndarray]:
    """Slice audio into overlapping windows.

    Returns a list of numpy arrays, each at most ``config.chunk_duration``
    seconds long, with stride ``config.stride``.
    """
    chunk_samples = int(config.chunk_duration * SAMPLE_RATE)
    stride_samples = int(config.stride * SAMPLE_RATE)
    total = len(audio)

    if total <= chunk_samples:
        return [audio]

    chunks: list[np.ndarray] = []
    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunks.append(audio[start:end])
        if end >= total:
            break
        start += stride_samples
    return chunks
