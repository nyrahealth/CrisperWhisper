"""Coverage-gated temperature fallback for collapsed chunk decodes.

Greedy decoding occasionally *collapses* on a chunk -- the model emits a
confident but near-empty transcription (e.g. ``"Meanwhile."`` for 30s of dense
speech).  This is a verbatim-mode failure mode that standard avg-logprob /
compression gates miss (the short output is high-confidence).  The reliable
signal is *coverage*: speech clearly fills the chunk but almost no words come
out.  When that happens we re-decode the chunk with escalating sampling
temperature (à la OpenAI Whisper's temperature fallback) and keep the first
result that covers the audio -- which preserves the verbatim style, unlike
falling back to the intended transcript.

Backend-agnostic: works on any engine exposing ``generate_sampled`` and
``cross_attention_for_tokens`` (CT2Engine and TransformersEngine both do).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from crisperwhisper.prompt import strip_prompt_artifacts

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol

# Tuned on the meanwhile benchmark: T=0.8 recovers all observed collapses;
# the ladder handles per-clip variation (some recover at 0.4, T=1.0 alone can
# re-break the easy ones).
FALLBACK_TEMPERATURES = (0.4, 0.6, 0.8, 1.0)
SAMPLES_PER_TEMP = 3       # recovery is stochastic -- several draws per temp so a
                           # collapse is reliably escaped (only runs on bad chunks)
SAMPLING_TOPK = 0          # 0 => sample from the full (temperature-scaled) softmax
MEL_FRAME_S = 0.01         # log-mel hop (10ms; twice the 20ms encoder frame)
MIN_SPEECH_SEC = 5.0       # mel pre-filter: only consider chunks with real speech
MIN_WORDS_PER_SEC = 0.5    # mel pre-filter: below this (vs speech sec) => suspect
                           # (real speech is ~2-4 wps, so this has wide margin)
DISCREPANCY_RATIO = 0.5    # collapse confirmed when words < ratio * sibling_words
MIN_REF_WORDS = 8          # ignore tiny reference counts (near-silent chunks)


def speech_active_seconds(mel: Optional[np.ndarray]) -> Optional[float]:
    """Estimate seconds of speech-active audio from a log-mel spectrogram.

    Uses a percentile energy gate: frames whose mean log-mel energy sits well
    above the per-chunk noise floor count as speech.  Returns ``None`` when no
    mel is available.
    """
    if mel is None:
        return None
    m = mel[0] if getattr(mel, "ndim", 2) == 3 else mel
    if m.size == 0:
        return 0.0
    energy = m.mean(axis=0).astype(np.float32)
    floor = float(np.percentile(energy, 10))
    peak = float(np.percentile(energy, 95))
    if peak - floor < 1e-3:
        return 0.0
    thr = floor + 0.2 * (peak - floor)
    active_frames = int(np.sum(energy > thr))
    return active_frames * MEL_FRAME_S


def is_undercovered(
    n_words: int,
    mel: Optional[np.ndarray],
    *,
    min_speech_sec: float = MIN_SPEECH_SEC,
    min_words_per_sec: float = MIN_WORDS_PER_SEC,
) -> bool:
    """True when speech fills the chunk but the decode produced too few words."""
    sec = speech_active_seconds(mel)
    if sec is None or sec < min_speech_sec:
        return False
    return n_words < min_words_per_sec * sec


def word_count(engine: EngineProtocol, gen_ids) -> int:
    """Number of (artifact-stripped) words a token sequence decodes to."""
    return len(strip_prompt_artifacts(
        engine.decode_tokens(gen_ids, skip_special=True)
    ).split())


def _discrepant(n_words: int, ref_words: int) -> bool:
    """Collapse confirmed: ref has real content but this decode is far shorter."""
    return ref_words >= MIN_REF_WORDS and n_words < DISCREPANCY_RATIO * ref_words


def decode_with_coverage_fallback(
    engine: EngineProtocol,
    features,
    mel: Optional[np.ndarray],
    prompt_tokens: list[int],
    gen_ids: list[int],
    attention: Optional[np.ndarray],
    *,
    max_length: int,
    want_attention: bool,
    enabled: bool = True,
    ref_word_count: Optional[int] = None,
    ref_prompt_tokens: Optional[list[int]] = None,
    temperatures=FALLBACK_TEMPERATURES,
    suppress_tokens: Optional[list[int]] = None,
) -> tuple[list[int], Optional[np.ndarray]]:
    """Re-decode a chunk with escalating temperature if it collapsed.

    Collapse detection (the trigger):

    * ``ref_word_count`` given (dual mode -- the sibling mode is already
      decoded): decide **immediately** on the mode discrepancy
      (``words < 0.5 * sibling_words``).  No mel, no extra decode.
    * else (single mode): a cheap **mel coverage pre-filter** flags suspect
      chunks; only then is the sibling mode (``ref_prompt_tokens``) decoded as
      a reference to **confirm** via the same discrepancy test.  If no sibling
      prompt is available, the mel pre-filter alone decides.

    ``gen_ids``/``attention`` are the base (greedy) decode.  Returns the same
    when not collapsed / disabled / unsupported; otherwise the best-covering
    sampled decode (attention recovered via a teacher-forced pass when
    ``want_attention``).
    """
    if not enabled or not hasattr(engine, "generate_sampled"):
        return gen_ids, attention

    base_n = word_count(engine, gen_ids)

    if ref_word_count is not None:
        # Dual: the sibling mode is in hand -- check the discrepancy now.
        if not _discrepant(base_n, ref_word_count):
            return gen_ids, attention
    else:
        # Single: cheap mel pre-filter, then confirm against a sibling decode.
        if not is_undercovered(base_n, mel):
            return gen_ids, attention
        if ref_prompt_tokens is not None:
            try:
                ref_ids = engine.generate(
                    features, [list(ref_prompt_tokens)], max_length=max_length,
                    suppress_tokens=suppress_tokens,
                )[0]
                if not _discrepant(base_n, word_count(engine, ref_ids)):
                    return gen_ids, attention  # sibling agrees it's short -> not a collapse
            except Exception:
                pass  # couldn't get reference -> trust the mel pre-filter

    # Lazy import to avoid a hard dependency cycle.
    from crisperwhisper.hallucination import find_token_loop

    best_ids, best_n = gen_ids, base_n
    best_is_sampled = False
    seed = 0
    done = False
    for temp in temperatures:
        if done:
            break
        for _ in range(SAMPLES_PER_TEMP):
            cur_seed = seed
            seed += 1
            try:
                sampled = engine.generate_sampled(
                    features, prompt_tokens,
                    max_length=max_length, temperature=float(temp),
                    topk=SAMPLING_TOPK, seed=cur_seed,
                    suppress_tokens=suppress_tokens,
                )
            except Exception:
                continue
            # Reject sampled runs that fell into a repetition loop.
            if find_token_loop(sampled, min_ngram=1, max_ngram=5) is not None:
                continue
            n = word_count(engine, sampled)
            if n > best_n:
                best_ids, best_n, best_is_sampled = sampled, n, True
            # Accept once the recovered decode clears the triggering gate.
            recovered = (
                not _discrepant(n, ref_word_count) if ref_word_count is not None
                else not is_undercovered(n, mel)
            )
            if recovered:
                best_ids, best_n, best_is_sampled = sampled, n, True
                done = True
                break

    if not best_is_sampled:
        return gen_ids, attention

    new_attention = attention
    if want_attention:
        new_attention = engine.cross_attention_for_tokens(
            features, prompt_tokens, best_ids,
        )
    return best_ids, new_attention
