"""Transcription result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenProbability:
    """One decoded token of a word and the probability the model gave it.

    ``probability`` is ``exp(log_softmax(raw_logits)[token_id])`` at the step
    that produced the token: the full-vocabulary softmax of the raw logits,
    before token suppression or repair bans are applied.
    """

    token: str
    token_id: int
    probability: float


@dataclass
class PhoneTimestamp:
    """A single phone with its start/end timestamps in seconds."""

    phone: str
    start: float
    end: float


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

    ``probability`` and ``tokens`` are set only with
    ``word_confidence=True``: ``probability`` is the mean of
    ``tokens[i].probability`` (length-fair across BPE splits; the product
    of those values is the joint probability of the word's tokens).

    ``timing_confidence``, ``kind`` and ``phones`` are set only for words
    timed by the forced aligner (``timing_method="forced_aligner"``):
    ``timing_confidence`` is the aligner's 0..1 acoustic fit of the word
    (low values point at transcript errors and words displaced by them),
    ``kind`` is ``"word"``, ``"filler"``, ``"sound"`` or ``"cutoff"``, and
    ``phones`` are the phone-level boundaries.  A word the aligner could
    not place keeps its attention timing and leaves these ``None``.
    """

    word: str
    start: float | None
    end: float | None
    probability: float | None = None
    tokens: list[TokenProbability] | None = None
    timing_confidence: float | None = None
    kind: str | None = None
    phones: list[PhoneTimestamp] | None = None


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
    mode: str  # "verbatim" | "intended" | "verbatimize" | "forced_align"
    duration: float  # audio duration in seconds
    processing_time: float  # wall-clock inference time in seconds
    chunks: list[ChunkResult] | None = None  # per-chunk details for longform
    words: list[WordTimestamp] | None = None  # word-level timestamps (v1)
    # how ``words`` were timed: "attention" | "forced_aligner"; None when
    # there are no word timestamps
    timing_method: str | None = None
