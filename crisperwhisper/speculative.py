"""Speculative decoding for CrisperWhisper with KV-cache persistence.

Uses a smaller/faster draft model to propose candidate tokens, then
the main model verifies them in a single batched forward pass.

Two verification modes are supported (controlled by ``semantic_mode``):

- **strict** (default): token-for-token match — output is *identical*
  to what the main model would produce on its own.
- **semantic**: accepts draft tokens whose accumulated decoded text
  matches the main model after normalisation (lowercase, strip
  punctuation).  Handles capitalization, punctuation, and different
  BPE splits of the same word.  Output words are guaranteed to match;
  formatting (punct/casing) follows the draft model.

The speedup comes from:
  1. Draft model is smaller -> fast per-token generation via ``forward_step_greedy``
  2. Main model verifies K candidates in one ``forward_batch_greedy`` call
  3. KV-cache is persisted across iterations -> no redundant recomputation
  4. On partial accept, main-model KV cache is rolled back via ``truncate_to_step``
     (only the draft model needs re-prefilling)

Requires a custom CTranslate2 build that exposes ``prefill``,
``forward_step_greedy``, ``forward_batch_greedy``, and
``truncate_to_step`` on the Whisper model.

Usage::

    model = CrisperWhisperModel(
        "nyrahealth/CrisperWhisper",
        draft_model="nyrahealth/CrisperWhisper-draft",
    )
    # strict (default): exact same output as main model alone
    result = model.transcribe("audio.wav", speculative_decoding=True)

    # semantic: same words, relaxed punct/casing — higher acceptance rate
    result = model.transcribe(
        "audio.wav", speculative_decoding=True, speculative_mode="semantic",
    )
"""

from __future__ import annotations

import atexit
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal

import ctranslate2
import numpy as np

from crisperwhisper import check_speculative_support
from crisperwhisper.hallucination import (
    DEFAULT_REPAIR_THRESHOLDS,
    find_token_loop,
)

if TYPE_CHECKING:
    from crisperwhisper.engine import CT2Engine

logger = logging.getLogger(__name__)

check_speculative_support()

SpeculativeMode = Literal["strict", "semantic"]

# ---------------------------------------------------------------------------
# Shared worker for overlapping the main and draft models.
#
# The main and draft are separate ``ctranslate2.models.Whisper`` instances,
# each with its own thread-pool worker and (since CT2 assigns one CUDA stream
# per worker thread) its own CUDA stream.  The encode/prefill bindings release
# the GIL while they block, so submitting the draft call on a second Python
# thread lets the two models' kernels run concurrently on the GPU instead of
# back-to-back.  One worker is enough: the caller thread drives the main model
# while this worker drives the draft model.  A single process-wide pool avoids
# leaking a thread per (per-call) SpeculativeDecoder instance.
# ---------------------------------------------------------------------------

_DRAFT_EXECUTOR: ThreadPoolExecutor | None = None


def _draft_executor() -> ThreadPoolExecutor:
    global _DRAFT_EXECUTOR
    if _DRAFT_EXECUTOR is None:
        _DRAFT_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cw-spec-draft",
        )
        atexit.register(_DRAFT_EXECUTOR.shutdown, wait=False)
    return _DRAFT_EXECUTOR

# ---------------------------------------------------------------------------
# GPU -> CPU argmax helpers  (kept for prefill logits which still return SV)
# ---------------------------------------------------------------------------


def _to_fp32_cpu(logits: ctranslate2.StorageView) -> np.ndarray:
    return np.array(
        logits.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
    )


def _attn_mean_row(attention: ctranslate2.StorageView) -> np.ndarray:
    """Collapse a single-step cross-attention StorageView to a 1-D row.

    ``prefill_with_attention`` / ``forward_step*_with_attention`` return
    attention of shape ``[1, num_selected_heads, F_enc]`` on device.  We
    head-average and move to CPU, yielding ``[F_enc]`` float32.
    """
    arr = _to_fp32_cpu(attention)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0)


def _argmax(logits: ctranslate2.StorageView) -> int:
    """Greedy argmax for a single-position logits StorageView on GPU."""
    arr = _to_fp32_cpu(logits)
    return int(arr.reshape(-1, arr.shape[-1])[0].argmax())


def _batch_argmax(logits: ctranslate2.StorageView) -> list[int]:
    """Vectorised argmax over the sequence dimension of a batched logits SV.

    Expects shape ``[1, seq_len, vocab_size]``.  Returns ``seq_len`` ints.
    """
    arr = _to_fp32_cpu(logits)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.argmax(axis=-1).tolist()


def _build_suppress_mask(suppress_tokens: list[int]) -> np.ndarray:
    """Build a boolean mask for token suppression.

    Returns an int array of token IDs to suppress, suitable for
    advanced indexing (``logits[..., mask] = -inf``).
    """
    return np.array(suppress_tokens, dtype=np.intp)


def _argmax_suppressed(
    logits: ctranslate2.StorageView,
    suppress_ids: np.ndarray,
) -> int:
    """Argmax with token suppression (sets suppressed logits to -inf)."""
    arr = _to_fp32_cpu(logits).reshape(-1, logits.shape[-1])[0]
    arr[suppress_ids] = -np.inf
    return int(arr.argmax())


def _batch_argmax_suppressed(
    logits: ctranslate2.StorageView,
    suppress_ids: np.ndarray,
) -> list[int]:
    """Batch argmax with token suppression."""
    arr = _to_fp32_cpu(logits)
    if arr.ndim == 3:
        arr = arr[0]
    arr[:, suppress_ids] = -np.inf
    return arr.argmax(axis=-1).tolist()


# ---------------------------------------------------------------------------
# Text normalisation for semantic verification
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:['.][a-zA-Z0-9]+)*")


def _normalize_text(text: str) -> str:
    """Extract lowercase alphanumeric words from *text*.

    Strips punctuation and whitespace, lowercases everything, but
    preserves in-word apostrophes (``don't``) and decimal points
    (``1.5``).  Returns a single-space-separated word string suitable
    for prefix comparison.
    """
    return " ".join(w.lower() for w in _WORD_RE.findall(text))


def _semantic_accept_count(
    candidates: list[int],
    verify_preds: list[int],
    eot_id: int,
    decode_fn,
) -> int:
    """Return how many draft tokens are *semantically* accepted.

    Walks through ``candidates`` (draft proposals, main-token-space)
    and ``verify_preds`` (main-model predictions) simultaneously,
    accumulating decoded text for both.  At each position:

    * **Normalised exact match**: the accumulated decoded texts,
      after lowercasing and stripping punctuation, are identical
      → confirmed (handles punct / case / BPE-split differences).
    * **Prefix relationship**: one normalised text is a prefix of the
      other → tentatively compatible (BPE split mid-word); continue
      but don't confirm yet.
    * **Neither**: real word-level divergence → stop.

    Returns the index *after* the last confirmed position (i.e. the
    number of draft tokens to accept).  Zero means "reject all".
    """
    last_confirmed = 0
    draft_text = ""
    main_text = ""
    texts_diverged = False

    for j in range(min(len(candidates), len(verify_preds))):
        if verify_preds[j] == candidates[j]:
            tok_text = decode_fn([candidates[j]])
            draft_text += tok_text
            main_text += tok_text
            if not texts_diverged:
                last_confirmed = j + 1
                if candidates[j] == eot_id:
                    break
                continue
        else:
            draft_text += decode_fn([candidates[j]])
            main_text += decode_fn([verify_preds[j]])
            texts_diverged = True

        d_norm = _normalize_text(draft_text)
        m_norm = _normalize_text(main_text)

        if d_norm == m_norm:
            last_confirmed = j + 1
        elif d_norm.startswith(m_norm) or m_norm.startswith(d_norm):
            pass  # tentatively compatible — BPE boundary mid-word
        else:
            break

        if candidates[j] == eot_id:
            break

    return last_confirmed


# ---------------------------------------------------------------------------
# Token ID remapping between main and draft vocabularies
# ---------------------------------------------------------------------------

_UNMAPPED = -1


class TokenRemapper:
    """Bidirectional token-ID translation between two Whisper vocabularies.

    Whisper model variants can have different numbers of language tokens
    which shifts all IDs above a threshold (task tokens, timestamps,
    custom verbatim/disfluency tags).  This class builds a string-based
    mapping so speculative decoding can translate between the two ID
    spaces without silent mismatches.
    """

    def __init__(self, main_engine: CT2Engine, draft_engine: CT2Engine):
        main_vocab: dict[str, int] = main_engine.tokenizer.get_vocab()
        draft_vocab: dict[str, int] = draft_engine.tokenizer.get_vocab()

        max_main = max(main_vocab.values()) + 1
        max_draft = max(draft_vocab.values()) + 1

        # draft_id → main_id  (indexed by draft_id)
        self._d2m = np.full(max_draft, _UNMAPPED, dtype=np.int32)
        # main_id → draft_id  (indexed by main_id)
        self._m2d = np.full(max_main, _UNMAPPED, dtype=np.int32)

        n_mapped = 0
        n_identity = 0
        for token_str, draft_id in draft_vocab.items():
            main_id = main_vocab.get(token_str)
            if main_id is not None:
                self._d2m[draft_id] = main_id
                self._m2d[main_id] = draft_id
                n_mapped += 1
                if main_id == draft_id:
                    n_identity += 1

        self.is_identity = n_identity == n_mapped == len(draft_vocab)

        logger.info(
            "TokenRemapper: %d/%d tokens mapped (%d identity, %d shifted), "
            "is_identity=%s",
            n_mapped, len(draft_vocab), n_identity,
            n_mapped - n_identity, self.is_identity,
        )

    def to_main(self, draft_id: int) -> int:
        """Map a single draft token ID to main-model space.

        Returns ``_UNMAPPED`` (-1) if no mapping exists.
        """
        if self.is_identity:
            return draft_id
        if 0 <= draft_id < len(self._d2m):
            return int(self._d2m[draft_id])
        return _UNMAPPED

    def to_draft(self, main_id: int) -> int:
        """Map a single main token ID to draft-model space.

        Returns ``_UNMAPPED`` (-1) if no mapping exists.
        """
        if self.is_identity:
            return main_id
        if 0 <= main_id < len(self._m2d):
            return int(self._m2d[main_id])
        return _UNMAPPED

    def prompt_to_draft(self, main_prompt: list[int]) -> list[int]:
        """Translate an entire prompt from main-model IDs to draft IDs."""
        if self.is_identity:
            return main_prompt
        out = []
        for t in main_prompt:
            d = self.to_draft(t)
            if d == _UNMAPPED:
                logger.warning(
                    "Prompt token %d has no draft mapping — keeping as-is", t,
                )
                out.append(t)
            else:
                out.append(d)
        return out

    def accepted_to_draft(self, accepted: list[int]) -> list[int]:
        """Translate accepted token IDs (main space) to draft space."""
        if self.is_identity:
            return accepted
        return [self.to_draft(t) for t in accepted]


class SpeculativeDecoder:
    """Wraps a main + draft CT2Engine pair for speculative decoding.

    Exposes the same ``generate`` and ``extract_features`` interface as
    ``CT2Engine`` so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        main_engine: CT2Engine,
        draft_engine: CT2Engine,
        num_speculative_tokens: int = 5,
        mode: SpeculativeMode = "strict",
        min_speculative_tokens: int = 0,
        max_speculative_tokens: int = 0,
    ):
        self.main = main_engine
        self.draft = draft_engine
        self.num_speculative_tokens = num_speculative_tokens
        self._mode: SpeculativeMode = mode
        # Adaptive-K window.  When ``max_speculative_tokens`` exceeds
        # ``min_speculative_tokens`` the per-round draft length adapts to
        # recent acceptance (HF-style additive increase / decrease),
        # seeded at ``num_speculative_tokens``; otherwise K is fixed.
        self.min_speculative_tokens = min_speculative_tokens
        self.max_speculative_tokens = max_speculative_tokens

        # Adaptive-K controller dynamics (mirrors the native C++ loop): AIMD
        # +2/-1.  K climbs by ``k_up_step`` after ``k_up_after`` fully-accepted
        # rounds and backs off by ``k_down_step`` after ``k_down_after``
        # rejected rounds.  The +2/-1 default converges to the ~1/3-full-accept
        # equilibrium and is biased toward the cap when acceptance is high
        # (the wall-time optimum for large-v2+turbo), while still backing off
        # on low-acceptance audio; the ``k_*_after`` knobs only exist for
        # experimentation (see research/timing).
        self.k_up_after = 1
        self.k_down_after = 1
        self.k_up_step = 2.0
        self.k_down_step = 1.0
        # Carry the controller's K across chunks so it converges to the
        # per-file equilibrium (seed becomes irrelevant).  Enabled whenever
        # adaptive K is used; reset at the first chunk of each transcription.
        self._persist_k = True
        self._k_state: float | None = None
        self._k_runs: tuple[int, int] = (0, 0)
        # Set True to re-seed the persistent K on the next decode (first
        # chunk of a new audio); cleared after that decode.
        self._reset_adaptive_next = True

        # Optional per-call stats collection for controller tuning.
        self._collect_stats = False
        self._spec_stats_log: list[dict] = []

        self.tokenizer = main_engine.tokenizer
        self.all_special_ids = main_engine.all_special_ids
        self.eot_id = main_engine.eot_id
        self.default_suppress_tokens = main_engine.default_suppress_tokens
        self.decode_tokens = main_engine.decode_tokens
        self.encode_text = main_engine.encode_text
        self.model = main_engine.model

        self._remap = TokenRemapper(main_engine, draft_engine)
        self._draft_eot = self._remap.to_draft(self.eot_id)

        self._needs_separate_features = (
            main_engine.n_mels != draft_engine.n_mels
        )
        self._last_audio: np.ndarray | None = None
        self._draft_features: ctranslate2.StorageView | None = None
        self._attention_enabled = False

    def extract_features(self, audio: np.ndarray) -> ctranslate2.StorageView:
        """Compute mel features for the main model.

        When the draft model uses a different number of mel bins, the
        raw audio is cached so that draft features can be computed
        lazily inside ``generate``.
        """
        if self._needs_separate_features:
            self._last_audio = audio
            self._draft_features = None
        return self.main.extract_features(audio)

    def extract_features_with_mel(
        self, audio: np.ndarray
    ) -> tuple[ctranslate2.StorageView, np.ndarray]:
        """Like :meth:`extract_features` but also returns the raw mel
        (used by the word-timing path to derive blank probabilities).
        """
        if self._needs_separate_features:
            self._last_audio = audio
            self._draft_features = None
        return self.main.extract_features_with_mel(audio)

    # ------------------------------------------------------------------
    # Cross-attention configuration (word timings under speculative
    # decoding).  Heads are enabled on *both* models: the draft model
    # supplies attention for accepted draft tokens, the main model for
    # the always-verified token and verifier corrections (Option B).
    # ------------------------------------------------------------------

    def enable_attention(
        self,
        heads: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Enable cross-attention collection on both engines.

        ``heads`` (if given) selects the main-model alignment heads.  The
        draft model uses its own stored alignment heads; if it has none,
        it falls back to the main selection (clamped to the draft's
        decoder depth) so the two attention sources stay frame-compatible.
        """
        main_heads = self.main.enable_attention(heads)

        draft_heads = self.draft.default_alignment_heads
        if not draft_heads:
            n_layers = getattr(self.draft, "_num_decoder_layers", None)
            if n_layers:
                draft_heads = [
                    (l, h) for (l, h) in main_heads if l < n_layers
                ]
            if not draft_heads:
                draft_heads = list(main_heads)
            logger.warning(
                "Draft model has no stored alignment_heads; falling back to "
                "the main-model selection for draft-token timing.  Word "
                "timings on accepted draft tokens may be less accurate."
            )
        self.draft.enable_attention(draft_heads)
        self._attention_enabled = True
        return main_heads

    def disable_attention(self) -> None:
        self.main.disable_attention()
        self.draft.disable_attention()
        self._attention_enabled = False

    def _get_draft_features(
        self, main_features: ctranslate2.StorageView,
    ) -> ctranslate2.StorageView:
        """Return features suitable for the draft model's encoder."""
        if not self._needs_separate_features:
            return main_features
        if self._draft_features is not None:
            return self._draft_features
        if self._last_audio is None:
            raise RuntimeError(
                "Draft model needs separate mel features but no raw audio "
                "was cached.  Call extract_features() first."
            )
        self._draft_features = self.draft.extract_features(self._last_audio)
        return self._draft_features

    def generate_with_repair(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Speculative greedy decode with hallucination repair.

        Pass 1 runs the **speculative** loop (draft proposes, main verifies),
        so this is the entry point that the longform ``continuation`` strategy
        and the short-audio path use to actually benefit from the draft model.
        Loop repair (the rare case) re-decodes the offending tail on the main
        model, mirroring :func:`hallucination.generate_with_repair`.

        In ``strict`` mode the speculative Pass 1 is token-identical to plain
        main-model greedy, so the repaired output matches the non-speculative
        ``generate_with_repair`` byte-for-byte.
        """
        if self._can_use_cpp_loop():
            accepted = self._speculative_generate_cpp(
                features,
                list(prompt_tokens),
                max_length=max_length,
                num_speculative_tokens=self.num_speculative_tokens,
                suppress_tokens=suppress_tokens,
            )
        else:
            accepted = self._speculative_generate(
                features,
                list(prompt_tokens),
                max_length=max_length,
                num_speculative_tokens=self.num_speculative_tokens,
                suppress_tokens=suppress_tokens,
                semantic=self._mode == "semantic",
            )
        if not hallucination_mitigation:
            return accepted
        return self._repair_tokens(
            features, list(prompt_tokens), accepted, max_length=max_length,
            suppress_tokens=suppress_tokens,
        )

    def _repair_tokens(
        self,
        features: ctranslate2.StorageView,
        prompt: list[int],
        accepted: list[int],
        *,
        max_length: int,
        max_repairs: int = 3,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Detect repetition loops in a speculative result and re-decode the
        offending tail on the **main** model (loop-starter banned on the first
        step), matching :func:`hallucination.generate_with_repair`'s rewind /
        escape behaviour.  Repairs are rare, so the bulk of decoding stays
        speculative; only the repaired tails fall back to main greedy.
        """
        from crisperwhisper.hallucination import _argmax_with_bans

        eot = self.main.eot_id
        sup = self.main._resolve_suppress(suppress_tokens)
        sup_ids = np.array(sup, dtype=np.intp) if sup else None
        encoded = self.main.model.encode(features)

        def escape_and_continue(prefix: list[int], ban: set[int], max_new: int) -> list[int]:
            if max_new <= 0:
                return []
            state, logits = self.main.model.prefill(encoded, prefix)
            first = _argmax_with_bans(logits, sup_ids, ban)
            if first == eot:
                return [eot]
            out = [first]
            for _ in range(max_new - 1):
                logits = self.main.model.forward_step(state, out[-1])
                tok = _argmax_with_bans(logits, sup_ids, None)
                if tok == eot:
                    out.append(eot)
                    break
                out.append(tok)
            return out

        for attempt in range(1, max_repairs + 1):
            hit = find_token_loop(
                accepted, min_ngram=1, max_ngram=5, reps=DEFAULT_REPAIR_THRESHOLDS,
            )
            if hit is None:
                break
            loop_start, gram = hit
            keep_end = loop_start + len(gram)  # keep_reps = 1
            trimmed = accepted[:keep_end]
            remaining = max_length - len(trimmed)
            logger.info(
                "Speculative repair %d: %d-gram loop at pos %d, rewinding to "
                "%d tokens and re-decoding tail on main model (banning %d)",
                attempt, len(gram), loop_start, keep_end, gram[0],
            )
            tail = escape_and_continue(prompt + trimmed, {gram[0]}, remaining)
            accepted = trimmed + tail

        return accepted

    def generate(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
        num_speculative_tokens: int | None = None,
        speculative_mode: SpeculativeMode | None = None,
    ) -> list[list[int]]:
        """Speculative-decode one or more prompts against the same features.

        Parameters
        ----------
        num_speculative_tokens
            Override the instance default for this call only.
        speculative_mode
            ``"strict"`` (exact token match) or ``"semantic"`` (normalised
            word match).  *None* inherits the instance default.
        """
        K = (
            num_speculative_tokens
            if num_speculative_tokens is not None
            else self.num_speculative_tokens
        )
        mode = speculative_mode if speculative_mode is not None else self._mode
        use_cpp = self._can_use_cpp_loop(mode=mode)
        results = []
        for prompt in prompt_tokens:
            if use_cpp:
                ids = self._speculative_generate_cpp(
                    features,
                    prompt,
                    max_length=max_length,
                    num_speculative_tokens=K,
                    suppress_tokens=suppress_tokens,
                )
            else:
                ids = self._speculative_generate(
                    features,
                    prompt,
                    max_length=max_length,
                    num_speculative_tokens=K,
                    suppress_tokens=suppress_tokens,
                    semantic=mode == "semantic",
                )
            results.append(ids)
        return results

    def _encode_both(
        self,
        main_features: ctranslate2.StorageView,
        draft_features: ctranslate2.StorageView,
    ) -> tuple[ctranslate2.StorageView, ctranslate2.StorageView]:
        """Encode features with both models (once per audio chunk).

        Returns pre-encoded representations that ``prefill`` accepts
        directly, skipping redundant re-encoding on every re-prefill.

        The draft encode is dispatched to a worker thread so it overlaps
        the main encode on a separate CUDA stream (both bindings release
        the GIL), instead of running back-to-back.
        """
        draft_future = _draft_executor().submit(
            self.draft.model.encode, draft_features,
        )
        main_enc = self.main.model.encode(main_features)
        draft_enc = draft_future.result()
        return main_enc, draft_enc

    def _prefill_both(
        self,
        main_enc: ctranslate2.StorageView,
        draft_enc: ctranslate2.StorageView,
        main_prompt: list[int],
    ):
        """Prefill both decoders using pre-encoded features.

        ``main_prompt`` is in main-model token space; it is automatically
        remapped to draft space for the draft model.

        The draft prefill is dispatched to a worker thread so it overlaps
        the main prefill on a separate CUDA stream.
        """
        draft_prompt = self._remap.prompt_to_draft(main_prompt)
        draft_future = _draft_executor().submit(
            self.draft.model.prefill, draft_enc, draft_prompt,
        )
        main_state, main_logits = self.main.model.prefill(main_enc, main_prompt)
        draft_state, _ = draft_future.result()
        return main_state, draft_state, main_logits

    def _prefill_draft_only(
        self,
        draft_enc: ctranslate2.StorageView,
        main_prompt: list[int],
    ):
        """Re-prefill only the draft decoder (main KV cache is rolled back)."""
        draft_prompt = self._remap.prompt_to_draft(main_prompt)
        draft_state, _ = self.draft.model.prefill(draft_enc, draft_prompt)
        return draft_state

    def _can_use_cpp_loop(self, mode: SpeculativeMode | None = None) -> bool:
        """Whether the native C++ speculative loop can serve this request.

        The C++ port (``Whisper.generate_speculative``) covers the *strict*,
        no-attention path only — the hot path for both the short-audio and
        longform continuation strategies.  Semantic verification and the
        word-timing (attention) loop stay in Python; if the running
        CTranslate2 build predates the binding, we transparently fall back
        to the Python loop.
        """
        eff_mode = mode if mode is not None else self._mode
        return (
            eff_mode != "semantic"
            and not self._attention_enabled
            and hasattr(self.main.model, "generate_speculative")
        )

    def _speculative_generate_cpp(
        self,
        features: ctranslate2.StorageView,
        prompt: list[int],
        *,
        max_length: int,
        num_speculative_tokens: int,
        suppress_tokens: list[int] | None,
    ) -> list[int]:
        """Strict speculative decode driven entirely by the C++ loop.

        Token-identical to :meth:`_speculative_generate` (strict mode): the
        native loop replicates the same draft/verify/accept/rollback logic
        and on-device argmax, but runs inside a single CTranslate2 job so
        there is no per-token Python dispatch or GIL traffic.
        """
        draft_features = self._get_draft_features(features)
        sup = (
            suppress_tokens
            if suppress_tokens is not None
            else self.default_suppress_tokens
        )
        sup_list = [int(t) for t in sup] if sup else []

        if self._remap.is_identity:
            d2m: list[int] = []
            m2d: list[int] = []
        else:
            d2m = self._remap._d2m.tolist()
            m2d = self._remap._m2d.tolist()

        reset_state = self._reset_adaptive_next
        self._reset_adaptive_next = False
        return self.main.model.generate_speculative(
            self.draft.model,
            features,
            draft_features,
            prompt=list(prompt),
            num_speculative_tokens=num_speculative_tokens,
            max_length=max_length,
            eot_id=self.eot_id,
            suppress_tokens=sup_list,
            d2m=d2m,
            m2d=m2d,
            min_speculative_tokens=self.min_speculative_tokens,
            max_speculative_tokens=self.max_speculative_tokens,
            reset_adaptive_state=reset_state,
        )

    def _speculative_generate(
        self,
        features: ctranslate2.StorageView,
        prompt: list[int],
        *,
        max_length: int,
        num_speculative_tokens: int,
        suppress_tokens: list[int] | None,
        semantic: bool = False,
    ) -> list[int]:
        """Core speculative decoding loop with KV-cache persistence.

        All token IDs stored in ``accepted`` and ``main_next`` live in
        **main-model** token space.  Conversions to/from draft space
        happen at the boundary of draft-model calls via ``self._remap``.

        Uses ``forward_step_greedy`` / ``forward_batch_greedy`` to keep
        argmax on the GPU (no full-vocab logit transfer to Python).

        On partial accept, the main model's KV cache is rolled back via
        ``truncate_to_step`` (~0.1 ms) instead of a full re-prefill
        (~21 ms).  Only the draft model is re-prefilled.

        When *semantic* is ``True``, the verification step uses
        accumulated-text normalisation instead of strict token matching,
        forgiving punctuation, capitalisation, and BPE-split differences.
        """
        K = num_speculative_tokens
        remap = self._remap
        draft_features = self._get_draft_features(features)

        # Adaptive-K controller (mirrors the native C++ loop): additive
        # increase on a fully-accepted round, additive decrease on any
        # rejection, clamped to [k_lo, k_hi].  Disabled => fixed K.
        adaptive = (
            self.max_speculative_tokens > 0
            and self.max_speculative_tokens > self.min_speculative_tokens
        )
        k_lo = max(1, self.min_speculative_tokens)
        k_hi = max(k_lo, self.max_speculative_tokens)
        k_step_up, k_step_down = self.k_up_step, self.k_down_step
        up_after, down_after = self.k_up_after, self.k_down_after
        acc_run = 0  # consecutive fully-accepted rounds
        rej_run = 0  # consecutive rounds with a rejection
        k_cur = float(K)
        if adaptive:
            k_cur = min(max(k_cur, float(k_lo)), float(k_hi))
            # When state persists across chunks the controller keeps
            # converging toward the acceptance-driven equilibrium over the
            # whole file, so the per-chunk seed stops mattering.  The first
            # chunk of a new audio re-seeds (``_reset_adaptive_next``).
            if self._persist_k:
                if self._reset_adaptive_next:
                    self._reset_adaptive_next = False
                    self._k_state = None
                if self._k_state is not None:
                    k_cur = min(max(self._k_state, float(k_lo)), float(k_hi))
                    acc_run, rej_run = self._k_runs

        # Controller stats (backend-independent efficiency signal).
        st_rounds = 0
        st_rollbacks = 0
        st_reprefills = 0
        st_draft_steps = 0
        st_k_sum = 0.0

        # Resolve suppress_tokens: use the model default when None.
        sup = suppress_tokens if suppress_tokens is not None else self.default_suppress_tokens
        # ``sup_list`` (main-vocab IDs) is handed to the greedy C++ APIs,
        # which mask those logits to -inf on the device *before* the argmax.
        # Suppression therefore stays on the fast greedy path (no full-vocab
        # logits transfer to Python) and matches a non-speculative suppressed
        # decode token-for-token.  ``sup_ids`` is only used for the Python-side
        # argmax over the prompt-prefill logits, which are returned in full
        # regardless and happen at most once per chunk / re-prefill.
        sup_list = list(sup) if sup else []
        sup_ids = _build_suppress_mask(sup) if sup else None

        def _prefill_argmax(logits: ctranslate2.StorageView) -> int:
            return _argmax(logits) if sup_ids is None else _argmax_suppressed(logits, sup_ids)

        main_enc, draft_enc = self._encode_both(features, draft_features)

        main_state, draft_state, main_logits = self._prefill_both(
            main_enc, draft_enc, prompt,
        )
        main_next = _prefill_argmax(main_logits)

        accepted: list[int] = []
        full_prompt = list(prompt)

        _decode = (
            (lambda ids: self.main.decode_tokens(ids, skip_special=False))
            if semantic else None
        )

        while len(accepted) < max_length:
            if main_next == self.eot_id:
                accepted.append(main_next)
                break

            budget = max_length - len(accepted)
            if adaptive:
                k_round = int(k_cur + 0.5)
                k_round = min(max(k_round, k_lo), k_hi)
            else:
                k_round = K
            draft_n = min(k_round, budget - 1)
            if draft_n <= 0:
                accepted.append(main_next)
                break

            # --- Draft phase (always greedy, no suppression needed) ---
            candidates_main: list[int] = []
            candidates_draft: list[int] = []
            draft_tok = remap.to_draft(main_next)

            if draft_tok == _UNMAPPED:
                accepted.append(main_next)
                st_reprefills += 1
                main_state, draft_state, main_logits = self._prefill_both(
                    main_enc, draft_enc, full_prompt + accepted,
                )
                main_next = _prefill_argmax(main_logits)
                continue

            for _ in range(draft_n):
                draft_tok = self.draft.model.forward_step_greedy(
                    draft_state, draft_tok,
                )
                st_draft_steps += 1
                main_tok = remap.to_main(draft_tok)

                if main_tok == _UNMAPPED:
                    break

                candidates_draft.append(draft_tok)
                candidates_main.append(main_tok)

                if draft_tok == self._draft_eot:
                    break

            if not candidates_main:
                accepted.append(main_next)
                st_reprefills += 1
                main_state, draft_state, main_logits = self._prefill_both(
                    main_enc, draft_enc, full_prompt + accepted,
                )
                main_next = _prefill_argmax(main_logits)
                continue

            # --- Verify phase ---
            st_rounds += 1
            st_k_sum += k_round
            main_step_before_verify = main_state.current_step
            batch_tokens = [main_next] + candidates_main

            verify_preds = self.main.model.forward_batch_greedy(
                main_state, batch_tokens, sup_list,
            )

            if semantic:
                n_draft_accepted = _semantic_accept_count(
                    candidates_main,
                    verify_preds,
                    self.eot_id,
                    _decode,
                )
                if n_draft_accepted < len(candidates_main):
                    correction = verify_preds[n_draft_accepted]
                else:
                    correction = None
            else:
                n_draft_accepted = 0
                correction = None
                for j in range(len(candidates_main)):
                    if j >= len(verify_preds):
                        break
                    if verify_preds[j] == candidates_main[j]:
                        n_draft_accepted += 1
                        if candidates_main[j] == self.eot_id:
                            break
                    else:
                        correction = verify_preds[j]
                        break

            accepted.append(main_next)
            accepted.extend(candidates_main[:n_draft_accepted])
            if correction is not None:
                accepted.append(correction)

            if accepted and accepted[-1] == self.eot_id:
                break

            # --- State management ---
            all_accepted = (
                n_draft_accepted == len(candidates_main)
                and correction is None
            )

            # Adapt K for the next round based on this round's acceptance,
            # with optional hysteresis: only step after ``up_after`` /
            # ``down_after`` *consecutive* rounds of the same outcome.
            if adaptive:
                if all_accepted:
                    acc_run += 1
                    rej_run = 0
                    if acc_run >= up_after:
                        k_cur = min(float(k_hi), k_cur + k_step_up)
                        acc_run = 0
                else:
                    rej_run += 1
                    acc_run = 0
                    if rej_run >= down_after:
                        k_cur = max(float(k_lo), k_cur - k_step_down)
                        rej_run = 0

            if all_accepted:
                if candidates_draft:
                    self.draft.model.forward_step_greedy(
                        draft_state, candidates_draft[-1],
                    )
                    st_draft_steps += 1

                last_idx = len(candidates_main)
                if last_idx < len(verify_preds):
                    main_next = verify_preds[last_idx]
                else:
                    st_reprefills += 1
                    main_state, draft_state, main_logits = self._prefill_both(
                        main_enc, draft_enc, full_prompt + accepted,
                    )
                    main_next = _prefill_argmax(main_logits)
            else:
                st_rollbacks += 1
                rollback_to = main_step_before_verify + 1 + n_draft_accepted
                main_state.truncate_to_step(rollback_to)

                main_next = self.main.model.forward_step_greedy(
                    main_state, accepted[-1], sup_list,
                )

                draft_state = self._prefill_draft_only(
                    draft_enc, full_prompt + accepted,
                )

            logger.debug(
                "spec iter: mode=%s accepted=%d, draft_accepted=%d/%d, "
                "correction=%s, total=%d",
                "semantic" if semantic else "strict",
                1 + n_draft_accepted + (1 if correction else 0),
                n_draft_accepted, len(candidates_main),
                correction, len(accepted),
            )

        if adaptive and getattr(self, "_persist_k", False):
            self._k_state = k_cur
            self._k_runs = (acc_run, rej_run)

        if self._collect_stats:
            self._spec_stats_log.append({
                "tokens": len(accepted),
                "rounds": st_rounds,
                "rollbacks": st_rollbacks,
                "reprefills": st_reprefills,
                "draft_steps": st_draft_steps,
                "mean_k": (st_k_sum / st_rounds) if st_rounds else 0.0,
            })

        return accepted

    # ------------------------------------------------------------------
    # Word-timing variant (Option B): capture per-token cross-attention
    # alongside the accepted tokens.  Accepted *draft* tokens keep the
    # draft model's attention (captured while drafting); the always-
    # verified token and verifier *corrections* keep the main model's
    # attention (captured in the batched verify pass).  Rejected draft
    # tokens never contribute a row, so the attention matrix stays 1:1
    # with the returned token list even across rollbacks.
    # ------------------------------------------------------------------

    def generate_with_attention(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        num_speculative_tokens: int | None = None,
        speculative_mode: SpeculativeMode | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Speculative-decode a single prompt, returning ``(gen_ids,
        attention)`` where ``attention`` is a ``[len(gen_ids), F_enc]``
        float32 matrix of mean-over-heads post-softmax cross-attention
        (1-to-1 with ``gen_ids``).

        Used by the word-timestamps path.  ``enable_attention`` is called
        automatically if it has not been already.
        """
        if not self._attention_enabled:
            self.enable_attention()
        K = (
            num_speculative_tokens
            if num_speculative_tokens is not None
            else self.num_speculative_tokens
        )
        mode = speculative_mode if speculative_mode is not None else self._mode
        accepted, attn_rows = self._speculative_generate_with_attention(
            features,
            list(prompt_tokens),
            max_length=max_length,
            num_speculative_tokens=K,
            suppress_tokens=suppress_tokens,
            semantic=mode == "semantic",
        )

        # Hallucination repair: detect repetition loops and re-decode the
        # tail on the MAIN model (banning the loop-starter on the first
        # step), exactly like the non-speculative
        # ``generate_with_repair_and_attention`` path.  Rewound tokens'
        # attention rows are dropped; the re-decoded tail carries
        # main-model attention rows so the matrix stays 1:1 with the
        # tokens.  The bulk of decoding stays speculative; only the (rare)
        # repaired tails fall back to main greedy.
        if hallucination_mitigation:
            accepted, attn_rows = self._repair_with_main(
                features, list(prompt_tokens), accepted, attn_rows,
                max_length=max_length,
                suppress_tokens=suppress_tokens,
            )

        if not attn_rows:
            return accepted, np.zeros((0, 0), dtype=np.float32)
        attention = np.stack(attn_rows[: len(accepted)], axis=0).astype(np.float32)
        return accepted, attention

    def _repair_with_main(
        self,
        features: ctranslate2.StorageView,
        prompt: list[int],
        accepted: list[int],
        attn_rows: list[np.ndarray],
        *,
        max_length: int,
        suppress_tokens: list[int] | None,
        max_repairs: int = 3,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Repair repetition loops in a speculative result by re-decoding
        the tail on the main model with the loop-starter banned.

        Mirrors :func:`hallucination.generate_with_repair_and_attention`
        but seeds from the speculative output.  Returns the repaired
        ``(accepted, attn_rows)`` (still 1-to-1).
        """
        eot = self.main.eot_id
        sup = suppress_tokens if suppress_tokens is not None else self.main.default_suppress_tokens
        main_enc = self.main.model.encode(features)

        for attempt in range(1, max_repairs + 1):
            hit = find_token_loop(
                accepted, min_ngram=1, max_ngram=5, reps=DEFAULT_REPAIR_THRESHOLDS,
            )
            if hit is None:
                break
            loop_start, gram = hit
            n = len(gram)
            keep_end = loop_start + n  # keep_reps = 1
            accepted = accepted[:keep_end]
            attn_rows = attn_rows[:keep_end]

            remaining = max_length - len(accepted)
            if remaining <= 0:
                break

            logger.info(
                "Speculative repair %d: %d-gram loop at pos %d, rewinding "
                "to %d tokens and re-decoding tail on main model (banning "
                "token %d)",
                attempt, n, loop_start, keep_end, gram[0],
            )

            state, tail = self.main.generate_greedy_with_attention(
                main_enc, prompt + accepted,
                max_new_tokens=remaining,
                eot_id=eot,
                suppress_tokens=sup,
                ban_first_tokens=[gram[0]],
            )
            tail_attn = self.main.decode_attention(state)  # [len(tail), F]
            accepted = accepted + list(tail)
            for i in range(len(tail)):
                if i < tail_attn.shape[0]:
                    attn_rows.append(tail_attn[i].astype(np.float32))
                elif attn_rows:
                    attn_rows.append(np.zeros_like(attn_rows[-1]))

        return accepted, attn_rows

    def _speculative_generate_with_attention(
        self,
        features: ctranslate2.StorageView,
        prompt: list[int],
        *,
        max_length: int,
        num_speculative_tokens: int,
        suppress_tokens: list[int] | None,
        semantic: bool,
    ) -> tuple[list[int], list[np.ndarray]]:
        K = num_speculative_tokens
        remap = self._remap
        draft_features = self._get_draft_features(features)

        sup = suppress_tokens if suppress_tokens is not None else self.default_suppress_tokens
        sup_ids = _build_suppress_mask(sup) if sup else None
        use_greedy = sup_ids is None or len(sup_ids) == 0

        def main_argmax(logits: ctranslate2.StorageView) -> int:
            return _argmax(logits) if use_greedy else _argmax_suppressed(logits, sup_ids)

        def main_batch_argmax(logits: ctranslate2.StorageView) -> list[int]:
            return (
                _batch_argmax(logits) if use_greedy
                else _batch_argmax_suppressed(logits, sup_ids)
            )

        main_enc, draft_enc = self._encode_both(features, draft_features)

        # Prefill main with attention (its row is the first main_next's
        # attention); draft is prefilled normally (draft attention is only
        # captured for *generated* draft tokens).
        draft_prompt = remap.prompt_to_draft(prompt)
        draft_future = _draft_executor().submit(
            self.draft.model.prefill, draft_enc, draft_prompt,
        )
        main_state, main_logits, main_attn0 = self.main.prefill_with_attention(
            main_enc, prompt,
        )
        draft_state, _ = draft_future.result()
        main_next = main_argmax(main_logits)
        pending_main_row = _attn_mean_row(main_attn0)

        accepted: list[int] = []
        attn_rows: list[np.ndarray] = []
        full_prompt = list(prompt)

        _decode = (
            (lambda ids: self.main.decode_tokens(ids, skip_special=False))
            if semantic else None
        )

        def reprefill(curr_accepted: list[int]):
            nonlocal main_state, draft_state, main_next, pending_main_row
            draft_future = _draft_executor().submit(
                self._prefill_draft_only, draft_enc, full_prompt + curr_accepted,
            )
            main_state, logits, attn = self.main.prefill_with_attention(
                main_enc, full_prompt + curr_accepted,
            )
            draft_state = draft_future.result()
            main_next = main_argmax(logits)
            pending_main_row = _attn_mean_row(attn)

        while len(accepted) < max_length:
            if main_next == self.eot_id:
                accepted.append(main_next)
                attn_rows.append(pending_main_row)
                break

            budget = max_length - len(accepted)
            draft_n = min(K, budget - 1)
            if draft_n <= 0:
                accepted.append(main_next)
                attn_rows.append(pending_main_row)
                break

            # --- Draft phase: capture the draft model's attention row for
            #     each proposed candidate. ---
            candidates_main: list[int] = []
            candidates_draft: list[int] = []
            candidates_draft_rows: list[np.ndarray] = []
            draft_tok = remap.to_draft(main_next)

            if draft_tok == _UNMAPPED:
                accepted.append(main_next)
                attn_rows.append(pending_main_row)
                reprefill(accepted)
                continue

            for _ in range(draft_n):
                draft_tok, d_attn = self.draft.forward_step_greedy_with_attention(
                    draft_state, draft_tok,
                )
                main_tok = remap.to_main(draft_tok)
                if main_tok == _UNMAPPED:
                    break
                candidates_draft.append(draft_tok)
                candidates_main.append(main_tok)
                candidates_draft_rows.append(_attn_mean_row(d_attn))
                if draft_tok == self._draft_eot:
                    break

            if not candidates_main:
                accepted.append(main_next)
                attn_rows.append(pending_main_row)
                reprefill(accepted)
                continue

            # --- Verify phase: batched main pass that also yields the
            #     main model's per-position attention. ---
            main_step_before_verify = main_state.current_step
            batch_tokens = [main_next] + candidates_main
            verify_logits, batch_attn_sv = self.main.forward_batch_with_attention(
                main_state, batch_tokens,
            )
            verify_preds = main_batch_argmax(verify_logits)
            batch_attn = np.array(batch_attn_sv)  # [T, F_enc], head-averaged

            if semantic:
                n_draft_accepted = _semantic_accept_count(
                    candidates_main, verify_preds, self.eot_id, _decode,
                )
                correction = (
                    verify_preds[n_draft_accepted]
                    if n_draft_accepted < len(candidates_main) else None
                )
            else:
                n_draft_accepted = 0
                correction = None
                for j in range(len(candidates_main)):
                    if j >= len(verify_preds):
                        break
                    if verify_preds[j] == candidates_main[j]:
                        n_draft_accepted += 1
                        if candidates_main[j] == self.eot_id:
                            break
                    else:
                        correction = verify_preds[j]
                        break

            # Commit tokens + their attention rows (Option B sourcing).
            accepted.append(main_next)
            attn_rows.append(pending_main_row)  # main-model row
            for j in range(n_draft_accepted):
                accepted.append(candidates_main[j])
                attn_rows.append(candidates_draft_rows[j])  # draft-model row
            if correction is not None:
                accepted.append(correction)
                attn_rows.append(batch_attn[n_draft_accepted])  # main-model row

            if accepted and accepted[-1] == self.eot_id:
                break

            # --- State management + next main_next attention ---
            all_accepted = (
                n_draft_accepted == len(candidates_main) and correction is None
            )
            if all_accepted:
                if candidates_draft:
                    self.draft.model.forward_step_greedy(
                        draft_state, candidates_draft[-1],
                    )
                last_idx = len(candidates_main)
                if last_idx < len(verify_preds):
                    main_next = verify_preds[last_idx]
                    pending_main_row = batch_attn[last_idx]
                else:
                    reprefill(accepted)
            else:
                rollback_to = main_step_before_verify + 1 + n_draft_accepted
                main_state.truncate_to_step(rollback_to)
                step_logits, step_attn = self.main.forward_step_with_attention(
                    main_state, accepted[-1],
                )
                main_next = main_argmax(step_logits)
                pending_main_row = _attn_mean_row(step_attn)
                draft_state = self._prefill_draft_only(
                    draft_enc, full_prompt + accepted,
                )

        return accepted, attn_rows
