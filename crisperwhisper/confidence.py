"""Per-token log-probabilities for word confidences.

The values are the model's own: ``log_softmax(raw_logits)[token]`` at the step
that produced each token, over the full vocabulary and before token
suppression or repair bans are applied.  The greedy decode captures them
inline (CTranslate2 ``return_token_logprobs``, Transformers
``output_logits``); ids the decode did not produce itself -- coverage
fallback, early-EOT recovery, speculative decoding -- are scored with one
teacher-forced pass, which yields the same values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def teacher_forced_logprobs(engine, features, prompt_tokens, gen_ids) -> np.ndarray:
    """Score ``gen_ids`` with a teacher-forced pass.

    A :class:`~crisperwhisper.speculative.SpeculativeDecoder` is scored with
    its main (verifier) engine.
    """
    scorer = getattr(engine, "main", engine)
    return scorer.token_logprobs_for_tokens(features, list(prompt_tokens), list(gen_ids))


def decode_with_attention(
    engine,
    features,
    prompt_tokens: list[int],
    *,
    max_length: int,
    hallucination_mitigation: bool,
    suppress_tokens: Optional[list[int]],
    want_logprobs: bool,
) -> tuple[list[int], np.ndarray, Optional[np.ndarray]]:
    """Repair-aware decode with inline cross-attention, plus token log-probs.

    Returns ``(gen_ids, attention, logprobs)``; ``logprobs`` is ``None``
    unless ``want_logprobs``.
    """
    if hasattr(engine, "generate_with_attention"):
        # Speculative decoder: draft/main attention capture with built-in
        # post-hoc repair; accepted draft tokens have no main-model logits.
        gen_ids, attention = engine.generate_with_attention(
            features, prompt_tokens,
            max_length=max_length,
            hallucination_mitigation=hallucination_mitigation,
            suppress_tokens=suppress_tokens,
        )
        logprobs = (
            teacher_forced_logprobs(engine, features, prompt_tokens, gen_ids)
            if want_logprobs else None
        )
        return gen_ids, attention, logprobs
    if want_logprobs:
        return engine.generate_with_repair_and_attention(
            features, prompt_tokens,
            max_length=max_length,
            hallucination_mitigation=hallucination_mitigation,
            suppress_tokens=suppress_tokens,
            return_logprobs=True,
        )
    gen_ids, attention = engine.generate_with_repair_and_attention(
        features, prompt_tokens,
        max_length=max_length,
        hallucination_mitigation=hallucination_mitigation,
        suppress_tokens=suppress_tokens,
    )
    return gen_ids, attention, None


def logprobs_for_final_ids(
    engine,
    features,
    prompt_tokens: list[int],
    gen_ids: list[int],
    decoded_ids: list[int],
    logprobs: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Keep ``logprobs`` if ``gen_ids`` are still the decoded ids, else rescore.

    Coverage fallback and early-EOT recovery may replace the decode's ids.
    """
    if logprobs is None or list(gen_ids) == list(decoded_ids):
        return logprobs
    return teacher_forced_logprobs(engine, features, prompt_tokens, gen_ids)
