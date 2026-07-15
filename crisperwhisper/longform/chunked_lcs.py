"""Chunked word-level LCS longform strategy.

Chunks are transcribed independently (no context dependency, batchable)
and stitched by finding the longest common contiguous word subsequence
in the overlap region between adjacent chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from crisperwhisper.longform.base import LongformConfig, make_chunks
from crisperwhisper.prompt import strip_prompt_artifacts
from crisperwhisper.result import ChunkResult

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol
    from crisperwhisper.prompt import PromptBuilder

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


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
        prev_words = result_words
        next_words = raw_texts[i]

        overlap_sec = config.chunk_duration - config.stride
        est_overlap_words = max(int(overlap_sec * 4), 10)

        suffix = prev_words[-est_overlap_words:]
        prefix = next_words[:est_overlap_words]

        sa, sb, length = _longest_common_subsequence(suffix, prefix)

        if length > 0:
            cut_prev = len(prev_words) - len(suffix) + sa + length
            result_words = prev_words[:cut_prev]
            result_words.extend(next_words[sb + length:])
            chunk_results[i].stitch_lcs_length = length
            chunk_results[i].stitch_lcs_words = " ".join(prefix[sb:sb + length])
        else:
            result_words.extend(next_words)
            chunk_results[i].stitch_lcs_length = 0
            chunk_results[i].stitch_lcs_words = ""

    return " ".join(result_words), chunk_results
