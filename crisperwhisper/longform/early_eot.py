"""Recovery of a context-conditioned early end-of-text (EOT).

Some context-conditioned CrisperWhisper checkpoints (notably ``large_pro``) can
emit an over-confident EOT at a sentence-final *pause* when a continuation
context is present, truncating the rest of a chunk's audio.  On the default 30 s
/ 26 s geometry the collapsing chunk is often the *last* one, so the loss is
silent and unrecoverable by the overlap.  See ``DOCS.md`` for the full analysis.

This module implements a decode-time gate, applied per chunk at the greedy
decode's stop:

1. **Trigger** -- ``P(EOT) < stop_prob_threshold``.  A confident stop is left
   alone.  This is also what makes trailing silence/noise safe: the model stays
   confident at a true end even when loud non-speech follows, so the gate never
   fires there.
2. **Speech remains** -- ``gap`` = speech-active seconds after the last placed
   word ``>= tail_min``.  Mel energy alone cannot see *whether the decode
   reached* the speech; the last-word position (from cross-attention word
   timing) is what distinguishes a premature stop from a window-edge stop.
3. **Confident-termination guard** -- re-decode forcing past the premature EOT
   and accept the longer decode only if it then reaches a *confident* EOT
   (``P(EOT) >= confident_prob``).  A genuine collapse finds a new confident
   sentence-end when pushed past the pause; a hallucination-into-noise (incl. a
   long-phrase repetition loop) never does, so it is reverted.

The gate needs two engine capabilities (feature-detected via ``hasattr``):
``eot_probability`` (P(EOT) at a decode's stop) and ``greedy_stops_and_decode``
(greedy decode that suppresses EOT for the first ``min_new_tokens`` steps and
reports its stop confidence).  Both backends implement them; engines without
them (e.g. the speculative decoder) simply skip the gate.
"""

from __future__ import annotations

import logging

import numpy as np

from crisperwhisper.longform.base import EarlyEotConfig

logger = logging.getLogger(__name__)

MEL_FRAME_S = 0.01  # log-mel hop (10 ms)


def engine_supports_recovery(engine) -> bool:
    """True when *engine* exposes the primitives the gate needs."""
    return hasattr(engine, "eot_probability") and hasattr(
        engine, "greedy_stops_and_decode"
    )


def _speech_active_after(mel, start_s: float) -> float:
    """Speech-active seconds in *mel* after ``start_s``.

    Percentile energy gate, matching
    :func:`crisperwhisper.fallback.speech_active_seconds`: a frame counts as
    speech when its mean log-mel energy sits 20 % of the way from the 10th to the
    95th percentile of the chunk.  Trailing silence / quiet noise stays below the
    gate, so it contributes ~0 s.
    """
    if mel is None:
        return 0.0
    m = mel[0] if getattr(mel, "ndim", 2) == 3 else mel
    m = np.asarray(m)
    if m.size == 0:
        return 0.0
    energy = m.mean(axis=0).astype(np.float32)
    floor = float(np.percentile(energy, 10))
    peak = float(np.percentile(energy, 95))
    if peak - floor < 1e-3:
        return 0.0
    thr = floor + 0.2 * (peak - floor)
    lo = max(int(start_s / MEL_FRAME_S), 0)
    if lo >= energy.shape[0]:
        return 0.0
    return float(np.sum(energy[lo:] > thr)) * MEL_FRAME_S


def _last_word_end(word_ts) -> float | None:
    """End time (chunk-local seconds) of the last placeable word, or ``None``."""
    ends = [float(wt.end) for wt in word_ts if wt.end is not None]
    return max(ends) if ends else None


def _content_len(gen_ids, eot_id) -> int:
    """Number of generated tokens excluding any trailing EOT(s)."""
    n = len(gen_ids)
    while n > 0 and eot_id is not None and gen_ids[n - 1] == eot_id:
        n -= 1
    return n


def recover_early_eot(
    engine,
    features,
    mel,
    prompt_tokens,
    gen_ids,
    *,
    word_ts,
    is_last,
    max_length,
    suppress_tokens,
    config: EarlyEotConfig,
):
    """Return token ids for the chunk, extending *gen_ids* past a premature EOT
    when (and only when) all three gate conditions hold.

    The conditions are checked cheapest-first, so healthy chunks are dismissed
    without a forward pass: the trailing-speech ``gap`` (from mel energy + the
    already-computed word timings) is tested *before* ``P(EOT)`` is measured, and
    the recovery re-decode runs last -- only on a genuine collapse candidate.

    Parameters
    ----------
    engine
        Inference engine exposing ``eot_probability`` + ``greedy_stops_and_decode``
        (and ``eot_id``).
    features, mel
        The chunk's encoder features and log-mel spectrogram.
    prompt_tokens
        The decoder prompt used for this chunk (mode tags + continuation ctx).
    gen_ids
        The chunk's current greedy decode (may include a trailing EOT).
    word_ts
        Per-word timings of *gen_ids* (1-to-1, unplaceable words carry
        ``end=None``) -- used to locate the last transcribed word.
    is_last
        Whether this is the final chunk (uses the smaller tail floor).
    config
        :class:`~crisperwhisper.longform.base.EarlyEotConfig`.

    Returns
    -------
    list[int]
        ``gen_ids`` unchanged, or a longer token sequence (trailing EOT
        appended) when a confident recovery is found.
    """
    if not config.enabled:
        return gen_ids

    # (1) cheap gap check first -- dismisses healthy chunks with no forward pass.
    last_end = _last_word_end(word_ts)
    if last_end is None:
        return gen_ids  # no placeable word -> cannot locate the decode position
    tail_min = config.tail_min_final if is_last else config.tail_min_nonfinal
    gap = _speech_active_after(mel, last_end)
    if gap < tail_min:
        return gen_ids  # no speech-active audio left to recover

    # (2) only now pay for P(EOT): a confident stop is left alone (this is also
    # what makes loud trailing non-speech safe -- the model stays confident).
    stop_prob = engine.eot_probability(
        features, list(prompt_tokens), gen_ids, suppress_tokens=suppress_tokens,
    )
    if stop_prob is None or stop_prob >= config.stop_prob_threshold:
        return gen_ids

    # (3) force past the premature EOT; keep it only if it terminates confidently.
    eot_id = getattr(engine, "eot_id", None)
    content_len = _content_len(gen_ids, eot_id)
    ext_ids, ext_prob = engine.greedy_stops_and_decode(
        features,
        list(prompt_tokens),
        max_length=max_length,
        suppress_tokens=suppress_tokens,
        min_new_tokens=content_len + 1,  # force at least one token past the stop
    )
    # Confident-termination guard: keep the extension only if it found a new,
    # confident sentence-end and actually made progress.
    if (
        ext_prob is not None
        and ext_prob >= config.confident_prob
        and len(ext_ids) > content_len
    ):
        logger.info(
            "early-EOT recovery%s: stop_p=%.2f gap=%.1fs -> +%d tokens "
            "(new stop_p=%.2f)",
            " [final]" if is_last else "",
            stop_prob,
            gap,
            len(ext_ids) - content_len,
            ext_prob,
        )
        if eot_id is not None:
            return list(ext_ids) + [eot_id]
        return list(ext_ids)
    logger.debug(
        "early-EOT reverted: stop_p=%.2f gap=%.1fs continuation not confident "
        "(ext_p=%s)",
        stop_prob,
        gap,
        None if ext_prob is None else round(ext_prob, 2),
    )
    return gen_ids
