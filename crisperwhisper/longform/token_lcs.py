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
from crisperwhisper.result import ChunkResult, WordTimestamp
from crisperwhisper.word_timing import group_tokens_into_words, monotonize_words

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol
    from crisperwhisper.prompt import PromptBuilder

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def _merge_with_provenance(
    sequences: list[list[int]],
    special_ids: set[int],
) -> list[tuple[int, int, int]]:
    """Merge token sequences by iterative suffix-prefix alignment, keeping
    per-token provenance.

    Same alignment as the HuggingFace ``_find_longest_common_sequence`` used
    in the ASR pipeline; each merged entry is ``(token_id, chunk_idx,
    orig_token_idx)`` where ``orig_token_idx`` indexes into that chunk's
    *unfiltered* token sequence (so it can be mapped back onto per-chunk
    word segmentations / timings).
    """
    seq = [
        (t, 0, k) for k, t in enumerate(sequences[0]) if t not in special_ids
    ]
    for chunk_idx, new_seq_raw in enumerate(sequences[1:], start=1):
        new_seq = [
            (t, chunk_idx, k)
            for k, t in enumerate(new_seq_raw)
            if t not in special_ids
        ]
        best_idx = 0
        best_score = 0.0
        for i in range(1, len(new_seq) + 1):
            eps = i / 10_000.0
            matches = sum(
                a[0] == b[0] for a, b in zip(seq[-i:], new_seq[:i])
            )
            score = matches / i + eps
            if matches > 1 and score > best_score:
                best_idx = i
                best_score = score
        seq.extend(new_seq[best_idx:])
    return seq


def _find_longest_common_token_sequence(
    sequences: list[list[int]],
    special_ids: set[int],
) -> list[int]:
    """Merge token sequences by iterative suffix-prefix alignment.

    Re-implementation of the HuggingFace ``_find_longest_common_sequence``
    used in the ASR pipeline.
    """
    return [t for t, _, _ in _merge_with_provenance(sequences, special_ids)]


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


def token_lcs_transcribe_with_word_timestamps(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult], list[WordTimestamp]]:
    """Token-level LCS longform that also returns per-word timestamps in the
    coordinate system of the original audio.

    Each chunk decodes exactly as in :func:`token_lcs_transcribe` (plain
    greedy), then its cross-attention is recovered with one teacher-forced
    pass -- generated tokens (and thus the merged text) are identical
    whether or not timestamps are requested.

    Timing works through token provenance: the merge tracks, for every
    token it keeps, which chunk it came from (:func:`_merge_with_provenance`).
    The merged token sequence is segmented into words, and each merged word
    takes its ``start`` from the source-chunk word containing its first
    token and its ``end`` from the source-chunk word containing its last
    token, both lifted into global time by their chunk offsets.  In the
    common case (all of a word's tokens from one chunk) this is exactly that
    chunk's Viterbi timing for the word.  A final
    :func:`~crisperwhisper.word_timing.monotonize_words` pass keeps the
    timeline forward-going at merge seams.  Words whose source timing the
    Viterbi could not place are omitted from the timestamp list.
    """
    from crisperwhisper.word_timing import extract_word_timings

    engine.enable_attention(alignment_heads)

    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Token LCS longform (+timestamps): %.1fs -> %d chunk(s)",
        len(audio) / SAMPLE_RATE, n_chunks,
    )

    if mode == "verbatim":
        prompt_tokens = prompt_builder.verbatim(hotwords=hotwords)
    else:
        prompt_tokens = prompt_builder.intended(hotwords=hotwords)

    token_seqs: list[list[int]] = []
    chunk_word_ts: list[list[WordTimestamp]] = []
    tok2word: list[dict[int, int]] = []  # per chunk: orig token idx -> word idx
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
        token_seqs.append(gen_ids)

        attention = engine.cross_attention_for_tokens(
            features, prompt_tokens, gen_ids,
        )
        word_ts_local = extract_word_timings(
            engine, gen_ids, attention, mel,
            audio_duration_s=chunk_dur,
            keep_unplaceable=True,
        )
        chunk_word_ts.append(word_ts_local)

        # Same segmentation extract_word_timings uses internally: maps each
        # content token (by its position in ``gen_ids``) to its word index,
        # so provenance tags can be resolved to per-chunk word timings.
        tok_pieces = [engine.tokenizer.decode([t]) for t in gen_ids]
        word_token_indices, _ = group_tokens_into_words(gen_ids, tok_pieces)
        mapping: dict[int, int] = {}
        for w_idx, tok_idxs in enumerate(word_token_indices):
            for k in tok_idxs:
                mapping[k] = w_idx
        tok2word.append(mapping)

        raw = engine.decode_tokens(gen_ids, skip_special=True)
        raw = strip_prompt_artifacts(raw)

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=raw,
            is_last=(i == n_chunks - 1),
        ))

        logger.info(
            "Chunk %d/%d: %d tokens (%d timed words)",
            i + 1, n_chunks, len(gen_ids),
            sum(1 for wt in word_ts_local if wt.start is not None),
        )

    merged = _merge_with_provenance(token_seqs, engine.all_special_ids)
    merged_tokens = [t for t, _, _ in merged]
    text = engine.decode_tokens(merged_tokens, skip_special=True)
    text = strip_prompt_artifacts(text)

    # Segment the merged token sequence into words and time each word from
    # its source-chunk Viterbi alignment via the provenance tags.
    merged_pieces = [engine.tokenizer.decode([t]) for t in merged_tokens]
    merged_word_groups, merged_word_texts = group_tokens_into_words(
        merged_tokens, merged_pieces,
    )

    def _global_time(pos: int, which: str):
        _, chunk_idx, orig_idx = merged[pos]
        w_idx = tok2word[chunk_idx].get(orig_idx)
        if w_idx is None:
            return None
        wt = chunk_word_ts[chunk_idx][w_idx]
        val = wt.start if which == "start" else wt.end
        if val is None:
            return None
        return round(float(val) + chunk_idx * config.stride, 3)

    words: list[WordTimestamp] = []
    for group, word_text in zip(merged_word_groups, merged_word_texts):
        start = _global_time(group[0], "start")
        end = _global_time(group[-1], "end")
        if start is None or end is None:
            continue
        # A seam word can mix two chunks' timings; never let end precede start.
        words.append(WordTimestamp(word=word_text, start=start, end=max(start, end)))

    monotonize_words(words)

    return text, chunk_results, words
