"""Continuation-context longform strategy.

The model was trained with a context-continuation objective: given a prompt
``{mode_tags} <ctx> last_few_words <ectx>``, it outputs only the
continuation beyond those words.  Chunks are processed sequentially --
each chunk's prompt includes the last K confirmed words from previous chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from crisperwhisper.fallback import decode_with_coverage_fallback, word_count
from crisperwhisper.longform.base import LongformConfig, make_chunks
from crisperwhisper.longform.early_eot import (
    engine_supports_recovery,
    recover_early_eot,
)
from crisperwhisper.prompt import strip_prompt_artifacts
from crisperwhisper.result import ChunkResult, WordTimestamp
from crisperwhisper.word_timing import monotonize_words

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol
    from crisperwhisper.prompt import PromptBuilder

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def _overlap_drop_index(
    word_ts: list[WordTimestamp],
    *,
    stride_sec: float,
    drop_words: int,
    timestamp_aware_drop: bool,
    is_last: bool,
) -> int:
    """Return the index up to which words are confirmed; ``words[idx:]`` drop.

    ``word_ts`` must be 1-to-1 with the chunk's words (use
    :func:`~crisperwhisper.word_timing.extract_word_timings` with
    ``keep_unplaceable=True``); unplaceable words carry ``start=None``.

    * Final chunk -> keep everything (nothing is re-transcribed afterwards).
    * Legacy / no placeable timings -> drop the last ``drop_words`` words
      (fixed count).
    * Timestamp-aware -> a trailing word is eligible to drop only if its audio
      *starts* inside the overlap region ``[stride_sec, chunk_duration]`` (so
      the next window, which begins at ``stride_sec``, actually re-covers it).
      ``drop_words`` caps how many trailing words may be dropped.  Words that
      start before the overlap are always kept -- the next window cannot
      re-cover them, so dropping them would lose them.  The boundary is
      ``max(n - drop_words, first_overlap_index)``: if the last ``drop_words``
      words all start before the overlap, ``first_overlap_index == n`` wins and
      nothing is dropped.
    """
    n = len(word_ts)
    if is_last:
        return n
    legacy = max(n - max(drop_words, 0), 0)
    has_placeable = any(wt.start is not None for wt in word_ts)
    if not timestamp_aware_drop or not has_placeable:
        return legacy
    first_overlap = n
    for idx, wt in enumerate(word_ts):
        if wt.start is not None and float(wt.start) >= stride_sec:
            first_overlap = idx
            break
    return max(legacy, first_overlap)


def continuation_transcribe(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    hallucination_mitigation: bool = True,
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult]]:
    """Run continuation-context longform transcription.

    Returns the full transcript and per-chunk details.

    When ``config.timestamp_aware_drop`` is set (the default) this delegates to
    :func:`continuation_transcribe_with_word_timestamps` -- which captures
    cross-attention to time each chunk's words -- so the boundary word-drop can
    be made overlap-aware (a trailing word is only dropped when the next window
    actually re-covers it).  The word timestamps themselves are discarded here.
    Set ``config.timestamp_aware_drop=False`` for the legacy fixed-count drop
    with no attention capture.
    """
    if config.timestamp_aware_drop:
        text, chunk_results, _ = continuation_transcribe_with_word_timestamps(
            engine, prompt_builder, audio, config,
            mode=mode, hotwords=hotwords,
            hallucination_mitigation=hallucination_mitigation,
            alignment_heads=alignment_heads,
            suppress_tokens=suppress_tokens,
        )
        return text, chunk_results

    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Continuation longform: %.1fs -> %d chunk(s) (stride=%.0fs, overlap=%.0fs)",
        len(audio) / SAMPLE_RATE, n_chunks,
        config.stride, config.chunk_duration - config.stride,
    )

    confirmed_words: list[str] = []
    chunk_results: list[ChunkResult] = []

    for i, chunk in enumerate(chunks):
        is_last = i == n_chunks - 1
        start_sec = i * config.stride
        end_sec = start_sec + len(chunk) / SAMPLE_RATE

        if i == 0:
            context_text = None
        else:
            ctx_words = confirmed_words[-config.context_words:] if confirmed_words else []
            context_text = " ".join(ctx_words) if ctx_words else None

        if mode == "verbatim":
            prompt_tokens = prompt_builder.verbatim(hotwords=hotwords, context=context_text)
        else:
            prompt_tokens = prompt_builder.intended(hotwords=hotwords, context=context_text)

        features = engine.extract_features(chunk)

        gen_ids = engine.generate_with_repair(
            features, prompt_tokens,
            max_length=config.max_new_tokens,
            hallucination_mitigation=hallucination_mitigation,
            suppress_tokens=suppress_tokens,
        )

        raw = engine.decode_tokens(gen_ids, skip_special=True)
        raw = strip_prompt_artifacts(raw)
        words = raw.split()

        if not words:
            logger.warning("Chunk %d/%d produced empty output.", i + 1, n_chunks)
            chunk_results.append(ChunkResult(
                chunk_idx=i, start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2), text="",
                context=context_text, is_last=is_last,
            ))
            continue

        if is_last:
            safe_words = words
        else:
            safe_words = words[:-config.drop_words] if len(words) > config.drop_words else words

        confirmed_words.extend(safe_words)

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=" ".join(safe_words),
            context=context_text,
            is_last=is_last,
        ))

        logger.info(
            "Chunk %d/%d: %d words (%d confirmed)",
            i + 1, n_chunks, len(words), len(safe_words),
        )

    return " ".join(confirmed_words), chunk_results


def continuation_transcribe_with_word_timestamps(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    mode: str = "verbatim",
    hotwords: list[str] | None = None,
    hallucination_mitigation: bool = True,
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,
) -> tuple[str, list[ChunkResult], list[WordTimestamp]]:
    """Continuation longform that also returns per-word timestamps in the
    coordinate system of the original audio.

    The cross-attention path is configured once via
    ``engine.enable_attention(alignment_heads)``.  Each chunk:

    * Generates with cross-attention captured (using the repair-aware path
      when ``hallucination_mitigation=True``).
    * Runs Viterbi word alignment in chunk-local seconds via
      :func:`crisperwhisper.word_timing.extract_word_timings`.
    * Drops the trailing ``config.drop_words`` words on non-final chunks
      (mirrors the text-stitching logic so timings stay aligned with the
      confirmed text).
    * Adds ``chunk_start_sec = i * config.stride`` to each surviving
      word's start/end to lift it to global audio coordinates.

    A final pass clamps each word's ``start`` so that it never goes
    backwards from the previous word's ``end``.  This keeps the timeline
    monotonic at chunk seams even when Viterbi places a word slightly
    before its predecessor's tail.
    """
    from crisperwhisper.word_timing import extract_word_timings

    engine.enable_attention(alignment_heads)

    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)

    logger.info(
        "Continuation longform (+timestamps): %.1fs -> %d chunk(s) "
        "(stride=%.0fs, overlap=%.0fs)",
        len(audio) / SAMPLE_RATE, n_chunks,
        config.stride, config.chunk_duration - config.stride,
    )

    confirmed_words: list[str] = []
    chunk_results: list[ChunkResult] = []
    global_word_timestamps: list[WordTimestamp] = []

    for i, chunk in enumerate(chunks):
        is_last = i == n_chunks - 1
        start_sec = i * config.stride
        chunk_dur = len(chunk) / SAMPLE_RATE
        end_sec = start_sec + chunk_dur

        if i == 0:
            context_text = None
        else:
            ctx_words = confirmed_words[-config.context_words:] if confirmed_words else []
            context_text = " ".join(ctx_words) if ctx_words else None

        if mode == "verbatim":
            prompt_tokens = prompt_builder.verbatim(hotwords=hotwords, context=context_text)
            sibling_prompt = prompt_builder.intended(hotwords=hotwords, context=context_text)
        else:
            prompt_tokens = prompt_builder.intended(hotwords=hotwords, context=context_text)
            sibling_prompt = prompt_builder.verbatim(hotwords=hotwords, context=context_text)

        features, mel = engine.extract_features_with_mel(chunk)

        # Speculative engines capture draft/main attention via their own
        # Option B path (with built-in hallucination repair); plain
        # engines use generate_with_repair_and_attention.  Both return a
        # token list + a 1-to-1 ``[T, F_enc]`` attention matrix.
        if hasattr(engine, "generate_with_attention"):
            gen_ids, attention = engine.generate_with_attention(
                features, prompt_tokens,
                max_length=config.max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                suppress_tokens=suppress_tokens,
            )
        else:
            gen_ids, attention = engine.generate_with_repair_and_attention(
                features, prompt_tokens,
                max_length=config.max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                suppress_tokens=suppress_tokens,
            )

        # Recover collapsed chunks via escalating-temperature re-decode.  Single
        # mode: mel pre-filter, confirmed against the sibling-mode decode.
        gen_ids, attention = decode_with_coverage_fallback(
            engine, features, mel, prompt_tokens, gen_ids, attention,
            max_length=config.max_new_tokens, want_attention=True,
            enabled=config.temperature_fallback,
            ref_prompt_tokens=sibling_prompt,
            suppress_tokens=suppress_tokens,
        )

        # Extract chunk-local word timings, kept 1-to-1 with the word
        # segmentation (unplaceable words appear as ``start=None`` placeholders).
        # The timing pipeline's word segmentation is the single canonical word
        # source, so text and timings stay perfectly aligned for the drop.
        word_ts_local = extract_word_timings(
            engine, gen_ids, attention, mel,
            audio_duration_s=chunk_dur,
            keep_unplaceable=True,
        )

        # Recover a context-conditioned early EOT (large_pro truncates at a
        # sentence-final pause when context is present).  Only fires on a
        # low-confidence stop that left speech-active audio uncovered, and only
        # keeps the extension if it terminates confidently (see
        # :func:`recover_early_eot`).  Re-capture attention/timings on recovery.
        if config.early_eot.enabled and engine_supports_recovery(engine):
            recovered = recover_early_eot(
                engine, features, mel, prompt_tokens, gen_ids,
                word_ts=word_ts_local, is_last=is_last,
                max_length=config.max_new_tokens, suppress_tokens=suppress_tokens,
                config=config.early_eot,
            )
            if len(recovered) != len(gen_ids):
                gen_ids = recovered
                attention = engine.cross_attention_for_tokens(
                    features, prompt_tokens, gen_ids,
                )
                word_ts_local = extract_word_timings(
                    engine, gen_ids, attention, mel,
                    audio_duration_s=chunk_dur,
                    keep_unplaceable=True,
                )

        words = [wt.word for wt in word_ts_local]

        if not words:
            logger.warning("Chunk %d/%d produced empty output.", i + 1, n_chunks)
            chunk_results.append(ChunkResult(
                chunk_idx=i, start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2), text="",
                context=context_text, is_last=is_last,
            ))
            continue

        # Overlap-aware boundary: drop trailing words only when the next window
        # re-covers them (see :func:`_overlap_drop_index`).
        keep = _overlap_drop_index(
            word_ts_local,
            stride_sec=config.stride,
            drop_words=config.drop_words,
            timestamp_aware_drop=config.timestamp_aware_drop,
            is_last=is_last,
        )
        safe_words = words[:keep]
        safe_word_ts = word_ts_local[:keep]

        confirmed_words.extend(safe_words)

        # Lift chunk-local timings into global audio time (skip placeholders
        # for words the Viterbi could not place -- they have no timestamp).
        for wt in safe_word_ts:
            if wt.start is None or wt.end is None:
                continue
            global_word_timestamps.append(WordTimestamp(
                word=wt.word,
                start=round(float(wt.start) + start_sec, 3),
                end=round(float(wt.end) + start_sec, 3),
            ))

        chunk_results.append(ChunkResult(
            chunk_idx=i,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            text=" ".join(safe_words),
            context=context_text,
            is_last=is_last,
        ))

        logger.info(
            "Chunk %d/%d: %d words (%d confirmed, %d timed)",
            i + 1, n_chunks, len(words), len(safe_words),
            sum(1 for wt in safe_word_ts if wt.start is not None),
        )

    # Monotonize at chunk seams: each word's start can't precede the
    # previous word's end.  Keeps the timeline strictly forward-going
    # even when Viterbi places a chunk's first word slightly before the
    # previous chunk's last word ended.
    monotonize_words(global_word_timestamps)

    return " ".join(confirmed_words), chunk_results, global_word_timestamps


def _decode_chunk_multi(
    engine: EngineProtocol,
    features_single,
    mel,
    prompts: list[list[int]],
    *,
    max_length: int,
    hallucination_mitigation: bool,
    word_timestamps: bool,
    suppress_tokens: list[int] | None = None,
):
    """Decode several per-mode prompts on one shared audio chunk in a single
    batched pass.

    The prompts may differ in length -- each mode carries its own running
    continuation context.  The native ``generate_dual_greedy`` primitive
    equalises them with a short per-row "catch-up" (each shorter prompt
    advances with its own real greedy tokens until all rows line up), then
    decodes every row together in lockstep, capturing each row's word-timing
    cross-attention inline.  The result is token-for-token identical to
    decoding each prompt on its own, so *every* chunk batches -- not just the
    ones whose contexts happen to tokenize to the same length.

    Returns a list of ``(gen_ids, attention | None)`` aligned with
    ``prompts`` (attention is a ``[len(gen_ids), F_enc]`` matrix when
    ``word_timestamps`` is set).
    """
    gens, attns = engine.generate_dual_greedy(
        features_single, prompts,
        max_length=max_length,
        hallucination_mitigation=hallucination_mitigation,
        word_timestamps=word_timestamps,
        features_single=features_single,
        suppress_tokens=suppress_tokens,
    )
    results: list[tuple[list[int], object]] = []
    for i in range(len(prompts)):
        attention = attns[i] if (word_timestamps and attns is not None) else None
        results.append((gens[i], attention))
    return results


def continuation_transcribe_dual(
    engine: EngineProtocol,
    prompt_builder: PromptBuilder,
    audio: np.ndarray,
    config: LongformConfig,
    *,
    modes: tuple[str, ...],
    hotwords: list[str] | None = None,
    hallucination_mitigation: bool = True,
    word_timestamps: bool = False,
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,
) -> list[tuple[str, list[ChunkResult], list[WordTimestamp]]]:
    """Continuation longform for several modes at once (one shared decode).

    Chunks are processed in lockstep across modes.  Each chunk's per-mode
    prompts -- which carry each mode's own running continuation context --
    are decoded together in a single batched pass (see
    :func:`_decode_chunk_multi`); the native catch-up primitive equalises
    their differing context lengths so *every* chunk's autoregressive decode
    is shared.  The result is token-for-token identical to running
    :func:`continuation_transcribe` (or
    :func:`continuation_transcribe_with_word_timestamps`) once per mode.

    Returns one ``(text, chunk_results, words)`` tuple per mode, in order.
    """
    from crisperwhisper.word_timing import extract_word_timings

    # Word timings are needed either to surface to the caller or to drive the
    # overlap-aware boundary drop; capture cross-attention when either applies.
    capture_attn = word_timestamps or config.timestamp_aware_drop
    if capture_attn:
        engine.enable_attention(alignment_heads)

    chunks = make_chunks(audio, config)
    n_chunks = len(chunks)
    n_modes = len(modes)

    logger.info(
        "Continuation longform dual (%s): %.1fs -> %d chunk(s)",
        ",".join(modes), len(audio) / SAMPLE_RATE, n_chunks,
    )

    confirmed: list[list[str]] = [[] for _ in modes]
    chunk_results: list[list[ChunkResult]] = [[] for _ in modes]
    global_ts: list[list[WordTimestamp]] = [[] for _ in modes]

    for i, chunk in enumerate(chunks):
        is_last = i == n_chunks - 1
        start_sec = i * config.stride
        chunk_dur = len(chunk) / SAMPLE_RATE
        end_sec = start_sec + chunk_dur

        prompts: list[list[int]] = []
        contexts: list[str | None] = []
        for j, mode in enumerate(modes):
            if i == 0:
                ctx = None
            else:
                ctx_words = (
                    confirmed[j][-config.context_words:] if confirmed[j] else []
                )
                ctx = " ".join(ctx_words) if ctx_words else None
            contexts.append(ctx)
            if mode == "verbatim":
                prompts.append(
                    prompt_builder.verbatim(hotwords=hotwords, context=ctx)
                )
            else:
                prompts.append(
                    prompt_builder.intended(hotwords=hotwords, context=ctx)
                )

        features1, mel = engine.extract_features_with_mel(chunk)
        decoded = _decode_chunk_multi(
            engine, features1, mel, prompts,
            max_length=config.max_new_tokens,
            hallucination_mitigation=hallucination_mitigation,
            word_timestamps=capture_attn,
            suppress_tokens=suppress_tokens,
        )

        # Both modes are already decoded -- gate each row on the discrepancy
        # against its best sibling row right away (no mel, no extra decode).
        base_counts = [word_count(engine, decoded[j][0]) for j in range(n_modes)]

        for j, mode in enumerate(modes):
            gen_ids, attention = decoded[j]

            ref_n = max(
                (base_counts[k] for k in range(n_modes) if k != j), default=0,
            )
            gen_ids, attention = decode_with_coverage_fallback(
                engine, features1, mel, prompts[j], gen_ids, attention,
                max_length=config.max_new_tokens, want_attention=capture_attn,
                enabled=config.temperature_fallback,
                ref_word_count=ref_n,
                suppress_tokens=suppress_tokens,
            )

            # When attention was captured, the timing pipeline's word
            # segmentation is the canonical (1-to-1) word source; otherwise
            # fall back to the plain decoded text.
            word_ts_local: list[WordTimestamp] = []
            if capture_attn:
                word_ts_local = extract_word_timings(
                    engine, gen_ids, attention, mel,
                    audio_duration_s=chunk_dur,
                    keep_unplaceable=True,
                )
                # Recover a context-conditioned early EOT for this row (mirrors
                # the single-mode path).  The recovery re-decode is single-prompt,
                # so only a triggered row gives up batching -- and just for its
                # own recovery; healthy rows are dismissed by the cheap gap check.
                if config.early_eot.enabled and engine_supports_recovery(engine):
                    recovered = recover_early_eot(
                        engine, features1, mel, prompts[j], gen_ids,
                        word_ts=word_ts_local, is_last=is_last,
                        max_length=config.max_new_tokens,
                        suppress_tokens=suppress_tokens, config=config.early_eot,
                    )
                    if len(recovered) != len(gen_ids):
                        gen_ids = recovered
                        attention = engine.cross_attention_for_tokens(
                            features1, prompts[j], gen_ids,
                        )
                        word_ts_local = extract_word_timings(
                            engine, gen_ids, attention, mel,
                            audio_duration_s=chunk_dur,
                            keep_unplaceable=True,
                        )
                words = [wt.word for wt in word_ts_local]
            else:
                raw = strip_prompt_artifacts(
                    engine.decode_tokens(gen_ids, skip_special=True)
                )
                words = raw.split()

            if not words:
                logger.warning(
                    "Chunk %d/%d (%s) produced empty output.",
                    i + 1, n_chunks, mode,
                )
                chunk_results[j].append(ChunkResult(
                    chunk_idx=i, start_sec=round(start_sec, 2),
                    end_sec=round(end_sec, 2), text="",
                    context=contexts[j], is_last=is_last,
                ))
                continue

            # Overlap-aware boundary per mode (independent -- fixed stride means
            # no shared-seek coupling between modes).
            if capture_attn:
                keep = _overlap_drop_index(
                    word_ts_local,
                    stride_sec=config.stride,
                    drop_words=config.drop_words,
                    timestamp_aware_drop=config.timestamp_aware_drop,
                    is_last=is_last,
                )
            else:
                keep = (
                    len(words) if is_last
                    else max(len(words) - config.drop_words, 0)
                )
            safe_words = words[:keep]
            safe_word_ts = word_ts_local[:keep]

            confirmed[j].extend(safe_words)

            # Surface timestamps only when the caller asked for them; skip
            # placeholders for words the Viterbi could not place.
            if word_timestamps:
                for wt in safe_word_ts:
                    if wt.start is None or wt.end is None:
                        continue
                    global_ts[j].append(WordTimestamp(
                        word=wt.word,
                        start=round(float(wt.start) + start_sec, 3),
                        end=round(float(wt.end) + start_sec, 3),
                    ))

            chunk_results[j].append(ChunkResult(
                chunk_idx=i,
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                text=" ".join(safe_words),
                context=contexts[j],
                is_last=is_last,
            ))

    out: list[tuple[str, list[ChunkResult], list[WordTimestamp]]] = []
    for j in range(n_modes):
        if word_timestamps:
            monotonize_words(global_ts[j])
        out.append((" ".join(confirmed[j]), chunk_results[j], global_ts[j]))
    return out
