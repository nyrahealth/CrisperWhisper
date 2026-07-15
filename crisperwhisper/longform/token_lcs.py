"""Token-level LCS longform strategy (Distil-Whisper / HF pipeline style).

Chunks are transcribed independently and stitched at the *token* level
by finding the best alignment between the suffix of the accumulated
token sequence and the prefix of the new one.
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


def _find_longest_common_token_sequence(
    sequences: list[list[int]],
    special_ids: set[int],
) -> list[int]:
    """Merge token sequences by iterative suffix-prefix alignment.

    Re-implementation of the HuggingFace ``_find_longest_common_sequence``
    used in the ASR pipeline.
    """
    seq = [t for t in sequences[0] if t not in special_ids]
    for new_seq_raw in sequences[1:]:
        new_seq = [t for t in new_seq_raw if t not in special_ids]
        best_idx = 0
        best_score = 0.0
        for i in range(1, len(new_seq) + 1):
            eps = i / 10_000.0
            matches = sum(a == b for a, b in zip(seq[-i:], new_seq[:i]))
            score = matches / i + eps
            if matches > 1 and score > best_score:
                best_idx = i
                best_score = score
        seq.extend(new_seq[best_idx:])
    return seq


def token_lcs_transcribe(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult]]:
    """Run token-level LCS longform transcription.

    All chunks are transcribed independently, then stitched at the token level.
    """
    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Token LCS longform: %.1fs -> %d chunk(s)",
        len(audio) / SAMPLE_RATE, n_chunks,
    )

    if mode == "verbatim":
        prompt_tokens = prompt_builder.verbatim(hotwords=hotwords)
    else:
        prompt_tokens = prompt_builder.intended(hotwords=hotwords)

    token_seqs: list[list[int]] = []
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

        token_seqs.append(gen_ids)
        raw = engine.decode_tokens(gen_ids, skip_special=True)
        raw = strip_prompt_artifacts(raw)

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=raw,
            is_last=(i == n_chunks - 1),
        ))

        logger.info("Chunk %d/%d: %d tokens", i + 1, n_chunks, len(gen_ids))

    merged = _find_longest_common_token_sequence(token_seqs, engine.all_special_ids)
    text = engine.decode_tokens(merged, skip_special=True)
    text = strip_prompt_artifacts(text)

    return text, chunk_results
