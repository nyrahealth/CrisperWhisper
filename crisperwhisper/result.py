"""Transcription result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WordTimestamp:
    """A single word with its start/end timestamps in seconds.

    In public results (``TranscriptionResult.words``) ``start``/``end`` are
    always floats.  ``None`` timings occur only on the internal
    ``extract_word_timings(keep_unplaceable=True)`` path, where words the
    Viterbi aligner could not place are kept as placeholders so the list
    stays 1-to-1 with the word segmentation (used by the longform
    overlap-aware boundary drop); such placeholders are filtered out before
    results are returned.
    """

    word: str
    start: float | None
    end: float | None


@dataclass
class ChunkResult:
    """Per-chunk detail for longform transcription."""

    chunk_idx: int
    start_sec: float
    end_sec: float
    text: str
    context: str | None = None
    is_last: bool = False

    # Stitching metadata (for LCS strategies)
    stitch_lcs_length: int | None = None
    stitch_lcs_words: str | None = None


@dataclass
class TranscriptionResult:
    """Result of a CrisperWhisper transcription."""

    text: str
    language: str
    mode: str  # "verbatim" | "intended" | "verbatimize"
    duration: float  # audio duration in seconds
    processing_time: float  # wall-clock inference time in seconds
    chunks: list[ChunkResult] | None = None  # per-chunk details for longform
    words: list[WordTimestamp] | None = None  # word-level timestamps (v1)
