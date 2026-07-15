"""Hallucination mitigation for CrisperWhisper (CTranslate2 backend).

Strategies:

1. **Consecutive n-gram blocking** (``generate_with_blocking``) — step-by-
   step decoding via the CT2 fork's ``prefill`` / ``forward_step`` APIs.
   At each decoding step, if emitting the greedy token would create more
   than ``block_reps`` consecutive copies of any n-gram, that token is
   suppressed in the logits and the next-best token is selected instead.
   This is the direct equivalent of the HuggingFace
   ``ConsecutiveNgramBlocker`` LogitsProcessor.

2. **Context repair** (``generate_with_repair``) — greedy decoding runs
   freely with no constraints.  When ``detect_reps`` consecutive copies of
   an n-gram are found post-hoc, the output is rewound to ``keep_reps``
   copies, one escape token is forced (loop-starter banned), and free
   decoding resumes.  The excess repetitions are removed from the output.

3. **Post-hoc loop detection** (``find_token_loop``) — scans a finished
   token sequence for repeated n-gram patterns.  Used as a safety net
   (e.g. after speculative decoding where step-by-step blocking is not
   possible).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ctranslate2
import numpy as np

if TYPE_CHECKING:
    from crisperwhisper.engine import CT2Engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU ↔ CPU helpers (shared with speculative.py)
# ---------------------------------------------------------------------------

def _to_fp32_cpu(logits: ctranslate2.StorageView) -> np.ndarray:
    return np.array(
        logits.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
    )


def _argmax_with_bans(
    logits: ctranslate2.StorageView,
    suppress_ids: np.ndarray | None,
    ban_ids: set[int] | None,
) -> int:
    """Greedy argmax with static suppression + dynamic per-step bans."""
    arr = _to_fp32_cpu(logits).reshape(-1, logits.shape[-1])[0]
    if suppress_ids is not None and len(suppress_ids):
        arr[suppress_ids] = -np.inf
    if ban_ids:
        for t in ban_ids:
            arr[t] = -np.inf
    return int(arr.argmax())


# ---------------------------------------------------------------------------
# Loop detection (unchanged — used as a post-hoc safety net)
# ---------------------------------------------------------------------------

DEFAULT_REPAIR_THRESHOLDS: dict[int, int] = {1: 8, 2: 8, 3: 4, 4: 3, 5: 3}
"""Per-ngram-size repetition thresholds for ``generate_with_repair``.

Keys are n-gram sizes, values are the number of consecutive repetitions
that trigger a repair.  Larger n-grams use lower thresholds because
long repeated phrases are almost never genuine speech.
"""


def find_token_loop(
    ids: list[int],
    min_ngram: int = 1,
    max_ngram: int = 5,
    reps: int | dict[int, int] = 8,
) -> tuple[int, tuple[int, ...]] | None:
    """Find the earliest position where consecutive n-grams repeat.

    Parameters
    ----------
    ids : list[int]
        Token ID sequence to scan.
    min_ngram, max_ngram : int
        Range of n-gram sizes to check.
    reps : int or dict[int, int]
        If an ``int``, the same threshold is used for all n-gram sizes.
        If a ``dict``, maps ``{ngram_size: threshold}`` — n-gram sizes
        not in the dict are skipped.

    Returns ``(loop_start_index, ngram_tuple)`` or ``None``.
    """
    thresholds: dict[int, int]
    if isinstance(reps, int):
        thresholds = {n: reps for n in range(min_ngram, max_ngram + 1)}
    else:
        thresholds = reps

    for i in range(len(ids)):
        for n, needed_reps in thresholds.items():
            if n < min_ngram or n > max_ngram:
                continue
            needed = n * needed_reps
            if i + needed > len(ids):
                continue
            gram = tuple(ids[i : i + n])
            if all(
                tuple(ids[i + r * n : i + (r + 1) * n]) == gram
                for r in range(1, needed_reps)
            ):
                return i, gram
    return None


# ---------------------------------------------------------------------------
# Consecutive n-gram blocker — real-time, step-by-step
# ---------------------------------------------------------------------------

def _compute_bans(
    generated: list[int],
    thresholds: dict[int, int],
) -> set[int]:
    """Return token IDs that must be banned at the *next* step.

    For each (n, max_reps) in *thresholds*, if the tail of *generated*
    already contains ``max_reps`` consecutive copies of the same n-gram,
    the first token of that n-gram is banned to prevent a further copy.

    This is a direct port of ``ConsecutiveNgramBlocker.__call__`` from
    the HuggingFace ``repetition_blocker.py``.
    """
    bans: set[int] = set()
    for n, max_reps in thresholds.items():
        needed = n * max_reps
        if len(generated) < needed:
            continue
        gram = tuple(generated[-n:])
        tail = generated[-needed:]
        if all(
            tuple(tail[i * n : (i + 1) * n]) == gram
            for i in range(max_reps)
        ):
            bans.add(gram[0])
    return bans


def generate_with_blocking(
    engine: CT2Engine,
    features: ctranslate2.StorageView,
    prompt_tokens: list[int],
    *,
    max_length: int = 256,
    block_thresholds: dict[int, int] | None = None,
    suppress_tokens: list[int] | None = None,
) -> list[int]:
    """Greedy decoding with real-time consecutive n-gram blocking.

    Uses the CT2 fork's ``encode`` / ``prefill`` / ``forward_step``
    APIs for token-by-token generation.  At each step the greedy token
    is checked against ``block_thresholds``; if it would create too many
    consecutive copies of an n-gram, that token is suppressed in the
    logits and the next-best token is selected instead.

    This is functionally identical to running HuggingFace ``generate()``
    with a ``ConsecutiveNgramBlocker`` LogitsProcessor.

    Parameters
    ----------
    engine : CT2Engine
        The CTranslate2 engine (must be the fork with ``prefill`` /
        ``forward_step`` support).
    features : ctranslate2.StorageView
        Pre-computed mel features (from ``engine.extract_features``).
    prompt_tokens : list[int]
        Decoder prompt (mode tags + SOT + language + task + notimestamps).
    max_length : int
        Maximum number of generated tokens (excluding prompt).
    block_thresholds : dict[int, int] | None
        ``{ngram_size: max_consecutive_reps}`` — when the n-gram has been
        repeated ``max_consecutive_reps`` times, the token that would start
        the next copy is banned for that step.  Defaults to
        ``{1: 4, 2: 4, 3: 3, 4: 3, 5: 3}``.
    suppress_tokens : list[int] | None
        Static suppress list; ``None`` uses the engine's config default.

    Returns
    -------
    list[int]
        Generated token IDs (no prompt prefix).
    """
    if block_thresholds is None:
        block_thresholds = {1: 8, 2: 8, 3: 3, 4: 3, 5: 3}

    sup = engine._resolve_suppress(suppress_tokens)
    sup_ids = np.array(sup, dtype=np.intp) if sup else None
    eot = engine.eot_id

    # Encode + prefill
    encoded = engine.model.encode(features)
    state, logits = engine.model.prefill(encoded, prompt_tokens)

    # First token
    first = _argmax_with_bans(logits, sup_ids, None)
    if first == eot:
        return [eot]

    generated: list[int] = [first]

    for _ in range(max_length - 1):
        logits = engine.model.forward_step(state, generated[-1])

        # Check *before* picking: does the current tail already have
        # max_reps copies of some n-gram?  If so, ban its first token.
        bans = _compute_bans(generated, block_thresholds)
        token = _argmax_with_bans(logits, sup_ids, bans if bans else None)

        if bans:
            logger.debug(
                "Blocked tokens %s at step %d, picked %d",
                bans, len(generated), token,
            )

        if token == eot:
            generated.append(eot)
            break

        generated.append(token)

    return generated


# ---------------------------------------------------------------------------
# Context repair — rewind-and-escape
# ---------------------------------------------------------------------------

def generate_with_repair(
    engine: CT2Engine,
    features: ctranslate2.StorageView,
    prompt_tokens: list[int],
    *,
    max_length: int = 256,
    detect_reps: int | dict[int, int] | None = None,
    keep_reps: int = 1,
    max_ngram: int = 5,
    max_repairs: int = 3,
    suppress_tokens: list[int] | None = None,
) -> tuple[list[int], int]:
    """Greedy decoding with automatic loop detection and context repair.

    The model runs freely (no per-step constraints).  When a loop is
    detected, the output is rewound and the model gets one forced
    "escape" token, then continues freely again.

    Algorithm:

    1. Run greedy decoding freely (no blocker).
    2. Scan the output for consecutive n-gram repetitions exceeding
       the per-size thresholds.
    3. If found, trim to the loop onset + ``keep_reps`` copies.
    4. Re-prefill up to the trim point, generate **one** token with the
       loop-starting token banned.
    5. Continue free decoding for the remaining budget (sharing KV state).
    6. Go back to step 2.

    Parameters
    ----------
    engine : CT2Engine
        The CTranslate2 engine (fork with prefill/forward_step support).
    features : ctranslate2.StorageView
        Pre-computed mel features.
    prompt_tokens : list[int]
        Decoder prompt tokens.
    max_length : int
        Maximum generated tokens (excluding prompt).
    detect_reps : int, dict[int, int], or None
        Repetition thresholds that trigger a repair.

        - ``int`` — same threshold for all n-gram sizes 1 to ``max_ngram``.
        - ``dict`` — per-size thresholds ``{ngram_size: reps}``.
          N-gram sizes not in the dict are not monitored.
        - ``None`` — use ``DEFAULT_REPAIR_THRESHOLDS``.
    keep_reps : int
        How many copies of the looping n-gram to keep after rewind.
    max_ngram : int
        Largest n-gram size to scan for loops.
    max_repairs : int
        Give up after this many rewind cycles.
    suppress_tokens : list[int] | None
        Static suppress list; ``None`` uses the engine's config default.

    Returns
    -------
    (gen_ids, n_repairs)
        gen_ids: generated token IDs (no prompt prefix).
        n_repairs: number of repair cycles performed (0 = clean).
    """
    if detect_reps is None:
        detect_reps = DEFAULT_REPAIR_THRESHOLDS

    sup = engine._resolve_suppress(suppress_tokens)
    sup_ids = np.array(sup, dtype=np.intp) if sup else None
    eot = engine.eot_id

    encoded = engine.model.encode(features)

    def _greedy_from(prefix: list[int], max_new: int) -> list[int]:
        """Free greedy decoding from a full prefix (prompt + prior tokens)."""
        if max_new <= 0:
            return []
        state, logits = engine.model.prefill(encoded, prefix)
        first = _argmax_with_bans(logits, sup_ids, None)
        if first == eot:
            return [eot]
        out = [first]
        for _ in range(max_new - 1):
            logits = engine.model.forward_step(state, out[-1])
            tok = _argmax_with_bans(logits, sup_ids, None)
            if tok == eot:
                out.append(eot)
                break
            out.append(tok)
        return out

    def _escape_and_continue(
        prefix: list[int], ban: set[int], max_new: int,
    ) -> list[int]:
        """Re-prefill, force one escape token, then continue freely.

        Shares the KV-cache state between the escape step and the
        continuation so we don't re-prefill twice.
        """
        if max_new <= 0:
            return []
        state, logits = engine.model.prefill(encoded, prefix)
        escape = _argmax_with_bans(logits, sup_ids, ban)
        if escape == eot:
            return [eot]
        out = [escape]
        for _ in range(max_new - 1):
            logits = engine.model.forward_step(state, out[-1])
            tok = _argmax_with_bans(logits, sup_ids, None)
            if tok == eot:
                out.append(eot)
                break
            out.append(tok)
        return out

    # Pass 1: free generation
    generated = _greedy_from(prompt_tokens, max_length)

    for attempt in range(1, max_repairs + 1):
        hit = find_token_loop(
            generated, min_ngram=1, max_ngram=max_ngram, reps=detect_reps,
        )
        if hit is None:
            return generated, attempt - 1

        loop_start, gram = hit
        n = len(gram)
        keep_end = loop_start + n * keep_reps
        trimmed = generated[:keep_end]

        reps_for_log = detect_reps.get(n, "?") if isinstance(detect_reps, dict) else detect_reps
        logger.info(
            "Repair %d: %d-gram loop at pos %d (%s reps), "
            "rewinding to %d tokens, banning token %d",
            attempt, n, loop_start, reps_for_log, keep_end, gram[0],
        )

        remaining = max_length - len(trimmed)
        tail = _escape_and_continue(
            prompt_tokens + trimmed, {gram[0]}, remaining,
        )
        generated = trimmed + tail

    return generated, max_repairs


# ---------------------------------------------------------------------------
# Variant that also captures per-step cross-attention rows.
# ---------------------------------------------------------------------------

def generate_with_repair_and_attention(
    engine: CT2Engine,
    features: ctranslate2.StorageView,
    prompt_tokens: list[int],
    *,
    max_length: int = 256,
    detect_reps: int | dict[int, int] | None = None,
    keep_reps: int = 1,
    max_ngram: int = 5,
    max_repairs: int = 3,
    suppress_tokens: list[int] | None = None,
) -> tuple[list[int], np.ndarray, int]:
    """Greedy decoding with hallucination repair, also collecting the
    per-step cross-attention rows for word-timing extraction.

    Performance: each bulk decoding segment runs **entirely inside one
    CTranslate2 thread-pool job** via
    :meth:`CT2Engine.generate_greedy_with_attention`.  Python is only
    re-entered between segments (i.e. once per repair).  In the common
    case of zero repairs this is a single Python -> C++ round-trip for
    the whole utterance, with no per-step GPU-to-CPU logits copies.

    The returned attention matrix is 1-to-1 with the returned
    ``gen_ids`` list (one row per generated token, in the same order,
    where row ``k`` is the attention used to predict token ``k``):

    1. Pass 1 runs ``generate_greedy_with_attention`` over the full
       budget.  Each generated token's attention row is buffered on the
       GPU inside the returned ``state.collected_attention``.
    2. When a loop is detected we trim to ``trimmed = generated[:keep_end]``,
       commit the attention rows for ``trimmed`` to a CPU-side
       ``kept_rows`` buffer (one bulk GPU->CPU copy per row), then call
       ``generate_greedy_with_attention`` again with
       ``prompt + trimmed`` as the new prompt and the loop-starter token
       in ``ban_first_tokens``.  That call captures the escape token's
       attention row plus rows for the continuation in one shot.
    3. After every repair, ``kept_rows`` plus the new state's rows
       always reconstruct the attention for the current ``generated``.

    Requires ``engine.enable_attention(...)`` to have been called first.

    Returns
    -------
    (gen_ids, attention, n_repairs)
        ``gen_ids`` is the token id list, ``attention`` is a
        ``(len(gen_ids), F_enc)`` float32 array of mean-over-heads
        post-softmax cross-attention, and ``n_repairs`` is the number of
        rewind cycles performed.
    """
    if detect_reps is None:
        detect_reps = DEFAULT_REPAIR_THRESHOLDS

    eot = engine.eot_id
    suppress_tokens = engine._resolve_suppress(suppress_tokens)

    # Encode once and reuse for every bulk decode segment.
    encoded = engine.model.encode(features)

    # Invariant: ``kept_rows`` always corresponds 1:1 to the **committed**
    # prefix of ``generated`` (the part we won't rewind further).  The
    # current ``state.collected_attention`` covers
    # ``generated[len(kept_rows):]``.
    kept_rows: list[np.ndarray] = []

    def _finalize(state, generated_local: list[int]) -> np.ndarray:
        # Bulk-transfer the current state's rows in a single PCIe copy
        # (concat + head-mean happen on the device).  ``kept_rows`` are
        # already on CPU from prior repair-time drains.
        state_arr = engine.decode_attention(state)
        if state_arr.size == 0:
            if not kept_rows:
                return np.zeros((0, 0), dtype=np.float32)
            return np.stack(kept_rows[:len(generated_local)], axis=0)
        if not kept_rows:
            return state_arr[:len(generated_local)]
        all_rows = np.concatenate(
            [np.stack(kept_rows, axis=0), state_arr], axis=0,
        )
        return all_rows[:len(generated_local)]

    # --- Pass 1: bulk free decode (no bans) --------------------------
    state, generated = engine.generate_greedy_with_attention(
        encoded, prompt_tokens,
        max_new_tokens=max_length,
        suppress_tokens=suppress_tokens,
        ban_first_tokens=[],
    )

    # --- Repair loop --------------------------------------------------
    for attempt in range(1, max_repairs + 1):
        hit = find_token_loop(
            generated, min_ngram=1, max_ngram=max_ngram, reps=detect_reps,
        )
        if hit is None:
            return generated, _finalize(state, generated), attempt - 1

        loop_start, gram = hit
        n = len(gram)
        keep_end = loop_start + n * keep_reps
        trimmed = generated[:keep_end]

        reps_for_log = (
            detect_reps.get(n, "?") if isinstance(detect_reps, dict) else detect_reps
        )
        logger.info(
            "Repair %d: %d-gram loop at pos %d (%s reps), "
            "rewinding to %d tokens, banning token %d",
            attempt, n, loop_start, reps_for_log, keep_end, gram[0],
        )

        # Reconcile ``kept_rows`` with ``keep_end``.  Normally we just
        # need to commit additional rows from the current state; if the
        # detected loop straddles a previously committed boundary
        # (rare, but possible when kept tokens repeat with new tail
        # tokens) we trim ``kept_rows`` back instead.
        n_to_commit = keep_end - len(kept_rows)
        if n_to_commit > 0:
            # One bulk GPU->CPU transfer (with on-device concat +
            # head-mean) instead of ``n_to_commit`` per-row copies.
            state_arr = engine.decode_attention(state)
            limit = min(n_to_commit, state_arr.shape[0])
            for i in range(limit):
                kept_rows.append(state_arr[i].copy())
        elif n_to_commit < 0:
            kept_rows = kept_rows[:keep_end]

        remaining = max_length - len(trimmed)
        if remaining <= 0:
            generated = trimmed
            attn = (
                np.stack(kept_rows, axis=0)
                if kept_rows
                else np.zeros((0, 0), dtype=np.float32)
            )
            return generated, attn, attempt

        # Bulk re-decode of the tail, banning the loop starter on the
        # first emitted step only.  All argmax / suppression /
        # attention accumulation happens inside one C++ thread-pool job.
        state, tail = engine.generate_greedy_with_attention(
            encoded, prompt_tokens + trimmed,
            max_new_tokens=remaining,
            suppress_tokens=suppress_tokens,
            ban_first_tokens=[gram[0]],
        )
        generated = trimmed + tail

    return generated, _finalize(state, generated), max_repairs
