"""Shared longform transcription utilities: config and chunking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000


@dataclass
class EarlyEotConfig:
    """Decode-time recovery of a context-conditioned early end-of-text.

    Some context-conditioned CrisperWhisper checkpoints (notably ``large_pro``)
    can emit an over-confident EOT at a sentence-final *pause* when a
    continuation context is present, truncating the rest of a chunk's audio.
    When enabled, the continuation strategy detects such a premature stop and,
    guarded against hallucinating into trailing silence/noise, forces the decode
    past it.  See :mod:`crisperwhisper.longform.early_eot` and DOCS.md.

    * ``stop_prob_threshold`` -- a stop is *suspect* only when its P(EOT) is
      below this. Confident stops (real ends, incl. trailing silence/noise where
      the model stays confident) are never touched. Load-bearing.
    * ``confident_prob`` -- the recovery is accepted only if, forced past the
      premature stop, the decode then reaches an EOT at least this confident.
      Otherwise the original stop is kept (revert). Prevents forcing the decode
      into a repetition-loop / silence hallucination. Must be
      ``>= stop_prob_threshold``.
    * ``tail_min_final`` / ``tail_min_nonfinal`` -- minimum speech-active seconds
      that must remain after the last transcribed word for a recovery to be
      attempted. The non-final floor equals the chunk overlap (a loss inside the
      overlap is re-covered by the next window anyway); the final chunk uses a
      smaller floor since nothing re-covers it.
    """

    enabled: bool = True
    stop_prob_threshold: float = 0.7
    confident_prob: float = 0.9
    tail_min_final: float = 2.0
    tail_min_nonfinal: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 < self.stop_prob_threshold <= 1.0:
            raise ValueError("stop_prob_threshold must be in (0, 1].")
        if not 0.0 < self.confident_prob <= 1.0:
            raise ValueError("confident_prob must be in (0, 1].")
        if self.confident_prob < self.stop_prob_threshold:
            raise ValueError(
                "confident_prob must be >= stop_prob_threshold "
                f"(got {self.confident_prob} < {self.stop_prob_threshold})."
            )


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
    early_eot: EarlyEotConfig = field(default_factory=EarlyEotConfig)


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
