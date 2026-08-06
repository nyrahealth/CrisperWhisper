"""Chunked word-level LCS longform strategy.

Chunks are transcribed independently (no context dependency, batchable)
and stitched by finding the longest common contiguous word subsequence
in the overlap region between adjacent chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

from crisperwhisper.longform.base import LongformConfig, make_chunks
from crisperwhisper.prompt import strip_prompt_artifacts
from crisperwhisper.result import ChunkResult, WordTimestamp
from crisperwhisper.word_timing import monotonize_words

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol
    from crisperwhisper.prompt import PromptBuilder

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000

T = TypeVar("T")


def _longest_common_subsequence(
    seq_a: list[str], seq_b: list[str]
) -> tuple[int, int, int]:
    """Longest common *contiguous* word subsequence.

    Returns ``(start_a, start_b, length)``.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return 0, 0, 0

    best_len = 0
    best_i = 0
    best_j = 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if seq_a[i - 1].lower() == seq_b[j - 1].lower():
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_i = i - best_len
                    best_j = j - best_len
            else:
                curr[j] = 0
        prev = curr

    return best_i, best_j, best_len


def _stitch_step(
    prev_items: list[T],
    next_items: list[T],
    config: LongformConfig,
    word_of: Callable[[T], str],
) -> tuple[list[T], int, str]:
    """Merge one chunk's items into the accumulated list via overlap LCS.

    Generic over the item type (plain word strings, or
    :class:`WordTimestamp` objects via ``word_of``) so the timed and
    untimed paths share the exact same stitch decision.

    Returns ``(merged, lcs_length, lcs_words)``.
    """
    overlap_sec = config.chunk_duration - config.stride
    est_overlap_words = max(int(overlap_sec * 4), 10)

    suffix = prev_items[-est_overlap_words:]
    prefix = next_items[:est_overlap_words]

    sa, sb, length = _longest_common_subsequence(
        [word_of(w) for w in suffix], [word_of(w) for w in prefix],
    )

    if length > 0:
        cut_prev = len(prev_items) - len(suffix) + sa + length
        merged = prev_items[:cut_prev]
        merged.extend(next_items[sb + length:])
        lcs_words = " ".join(word_of(w) for w in prefix[sb:sb + length])
        return merged, length, lcs_words

    return prev_items + next_items, 0, ""


def chunked_lcs_transcribe(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult]]:
    """Run chunked word-level LCS longform transcription.

    All chunks are transcribed independently, then stitched via LCS at
    overlap boundaries.
    """
    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Chunked LCS longform: %.1fs -> %d chunk(s)",
        len(audio) / SAMPLE_RATE, n_chunks,
    )

    if mode == "verbatim":
        prompt_tokens = prompt_builder.verbatim(hotwords=hotwords)
    else:
        prompt_tokens = prompt_builder.intended(hotwords=hotwords)

    raw_texts: list[list[str]] = []
    chunk_results: list[ChunkResult] = []

    for i, chunk in enumerate(chunks):
        start_sec = i * config.stride
        end_sec = start_sec + len(chunk) / SAMPLE_RATE
        features = engine.extract_features(chunk)

        gen_ids = engine.generate(
            features, [prompt_tokens],
            max_length=config.max_new_tokens,
            suppress_tokens=suppress_tokens,
        )[0]

        raw = engine.decode_tokens(gen_ids, skip_special=True)
        raw = strip_prompt_artifacts(raw)
        words = raw.split()
        raw_texts.append(words)

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=raw,
            is_last=(i == n_chunks - 1),
        ))

        logger.info("Chunk %d/%d: %d words", i + 1, n_chunks, len(words))

    # Stitch via LCS
    result_words = list(raw_texts[0])
    for i in range(1, n_chunks):
        result_words, length, lcs_words = _stitch_step(
            result_words, raw_texts[i], config, word_of=lambda w: w,
        )
        chunk_results[i].stitch_lcs_length = length
        chunk_results[i].stitch_lcs_words = lcs_words

    return " ".join(result_words), chunk_results


def chunked_lcs_transcribe_with_word_timestamps(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult], list[WordTimestamp]]:
    """Chunked LCS longform that also returns per-word timestamps in the
    coordinate system of the original audio.

    Each chunk decodes exactly as in :func:`chunked_lcs_transcribe` (plain
    greedy, no repair), then its cross-attention is recovered with one
    teacher-forced pass (``engine.cross_attention_for_tokens``) -- so the
    generated tokens are identical whether or not timestamps are requested.
    Chunk-local Viterbi timings are lifted into global time by the chunk
    offset, the usual LCS stitch then operates on the timestamped words
    (same stitch decision as the untimed path -- it compares word strings),
    and a final :func:`~crisperwhisper.word_timing.monotonize_words` pass
    keeps the timeline forward-going at chunk seams.

    As in the continuation strategy, the timing pipeline's word segmentation
    is the canonical word source here, so text and timings stay 1-to-1.
    Words the Viterbi cannot place keep their position in the text but are
    omitted from the returned timestamp list.
    """
    from crisperwhisper.word_timing import extract_word_timings

    engine.enable_attention(alignment_heads)

    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Chunked LCS longform (+timestamps): %.1fs -> %d chunk(s)",
        len(audio) / SAMPLE_RATE, n_chunks,
    )

    if mode == "verbatim":
        prompt_tokens = prompt_builder.verbatim(hotwords=hotwords)
    else:
        prompt_tokens = prompt_builder.intended(hotwords=hotwords)

    per_chunk_ts: list[list[WordTimestamp]] = []
    chunk_results: list[ChunkResult] = []

    for i, chunk in enumerate(chunks):
        start_sec = i * config.stride
        chunk_dur = len(chunk) / SAMPLE_RATE
        end_sec = start_sec + chunk_dur
        features, mel = engine.extract_features_with_mel(chunk)

        gen_ids = engine.generate(
            features, [prompt_tokens],
            max_length=config.max_new_tokens,
            suppress_tokens=suppress_tokens,
        )[0]

        attention = engine.cross_attention_for_tokens(
            features, prompt_tokens, gen_ids,
        )
        word_ts_local = extract_word_timings(
            engine, gen_ids, attention, mel,
            audio_duration_s=chunk_dur,
            keep_unplaceable=True,
        )

        # Lift chunk-local timings into global audio time; unplaceable words
        # keep their position (and word text) with ``None`` timings.
        lifted = [
            WordTimestamp(
                word=wt.word,
                start=None if wt.start is None else round(float(wt.start) + start_sec, 3),
                end=None if wt.end is None else round(float(wt.end) + start_sec, 3),
            )
            for wt in word_ts_local
        ]
        per_chunk_ts.append(lifted)

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=" ".join(wt.word for wt in lifted),
            is_last=(i == n_chunks - 1),
        ))

        logger.info(
            "Chunk %d/%d: %d words (%d timed)",
            i + 1, n_chunks, len(lifted),
            sum(1 for wt in lifted if wt.start is not None),
        )

    # Stitch via LCS -- identical stitch decision to the untimed path (the
    # LCS compares word strings), but carrying each word's timestamp through.
    result: list[WordTimestamp] = list(per_chunk_ts[0])
    for i in range(1, n_chunks):
        result, length, lcs_words = _stitch_step(
            result, per_chunk_ts[i], config, word_of=lambda wt: wt.word,
        )
        chunk_results[i].stitch_lcs_length = length
        chunk_results[i].stitch_lcs_words = lcs_words

    text = " ".join(wt.word for wt in result)
    words = [wt for wt in result if wt.start is not None and wt.end is not None]
    monotonize_words(words)

    return text, chunk_results, words
