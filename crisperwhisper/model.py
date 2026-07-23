"""CrisperWhisperModel — the main public API for CrisperWhisper inference.

Supports both CrisperWhisper v1 (``nyrahealth/CrisperWhisper``) and v2
(``nyralabs/CrisperWhisper2.0_<size>``) models.  The backend is selected
automatically:

- **v1**: HuggingFace Transformers (verbatim only, word-level timestamps)
- **v2**: CTranslate2 with speculative decoding, longform, hotwords, etc.

V1 models emit a deprecation warning on load.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from crisperwhisper.audio import SAMPLE_RATE, get_duration, load_audio
from crisperwhisper.result import TranscriptionResult

logger = logging.getLogger(__name__)

SHORT_THRESHOLD_S = 30.0

# Official CrisperWhisper 2.0 checkpoints. Bare size names are accepted as
# shorthand everywhere a model id is expected: CrisperWhisperModel("turbo").
# The *_pro variants are Nyra's best models (commercial license only).
OFFICIAL_MODELS = {
    size: f"nyralabs/CrisperWhisper2.0_{size}"
    for size in (
        "large", "turbo", "medium", "small",
        "large_pro", "turbo_pro", "medium_pro", "small_pro",
    )
}
DEFAULT_MODEL = OFFICIAL_MODELS["large"]


def resolve_model_id(name_or_path: str) -> str:
    """Expand a size shorthand to its official model id; pass through others."""
    return OFFICIAL_MODELS.get(name_or_path, name_or_path)

_V1_DEPRECATION = (
    "You are using the legacy CrisperWhisper v1 model. It is deprecated and "
    "will NOT be supported in the next release. Please migrate to "
    "CrisperWhisper 2.0 (nyralabs/CrisperWhisper2.0_large), which has much better "
    "verbatim performance and also adds 3-5x faster inference, intended mode, "
    "hotwords, verbatimize, longform, and speculative decoding. "
    "See https://github.com/nyrahealth/CrisperWhisper for details."
)

_V1_FEATURE_WARNING = (
    "'{feature}' is only available with CrisperWhisper v2 models. "
    "This v1 model will produce a verbatim transcription without {feature}. "
    "Upgrade to nyrahealth/CrisperWhisper2 for full feature support."
)

_BACKEND_MODULES = {
    "ct2": ["ctranslate2"],
    "transformers": ["torch", "transformers"],
}


def _backend_available(backend: str) -> bool:
    import importlib.util

    return all(
        importlib.util.find_spec(m) is not None
        for m in _BACKEND_MODULES[backend]
    )


def _sanitize_suppress_tokens(
    suppress_tokens: list[int] | None,
) -> list[int] | None:
    """Normalize a user-supplied suppress list once at the API boundary.

    ``None`` keeps the model's generation_config default; an explicit list
    (possibly empty = "suppress nothing") is cleaned of negative ids (the HF
    ``-1`` "default set" sentinel) so every downstream decode path -- both
    backends, repair, speculative, fallback -- receives explicit ids only.
    """
    if suppress_tokens is None:
        return None
    return [int(t) for t in suppress_tokens if int(t) >= 0]


def _require_backend(backend: str) -> None:
    """Raise a helpful ImportError if a backend's extra is not installed."""
    if not _backend_available(backend):
        raise ImportError(
            f"The '{backend}' backend requires the crisperwhisper[{backend}] "
            f"extra (provides {', '.join(_BACKEND_MODULES[backend])}). "
            f"Install it with:\n  pip install crisperwhisper[{backend}]"
        )


class CrisperWhisperModel:
    """High-level interface for CrisperWhisper speech recognition.

    Automatically detects whether the model is v1 or v2 and selects the
    appropriate backend.

    Parameters
    ----------
    model_name_or_path
        HuggingFace model ID, local directory, or one of the size
        shorthands ``"large"``, ``"turbo"``, ``"medium"``, ``"small"``
        for the official ``nyralabs/CrisperWhisper2.0_<size>``
        checkpoints.  Defaults to ``nyralabs/CrisperWhisper2.0_large``.

        - ``"nyralabs/CrisperWhisper2.0_<size>"`` — v2, CTranslate2 or
          Transformers backend
        - ``"nyrahealth/CrisperWhisper"`` — v1, HF Transformers backend
          (deprecated)

    backend
        Which v2 inference backend to use: ``"ct2"`` (CTranslate2, fast,
        supports speculative decoding), ``"transformers"`` (pure torch, no
        speculative decoding), or ``"auto"`` (default: prefer ``ct2`` when
        ``ctranslate2`` is installed, else ``transformers``).  Ignored for
        v1 models, which always use the legacy HuggingFace pipeline.
    compute_type
        Quantization / dtype.  CTranslate2: ``"float16"``,
        ``"int8_float16"``, ``"int8"``, ``"float32"``.  Transformers maps
        these onto the matching torch dtype (``int8*`` -> ``float16``).
    device
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default).
    device_index
        GPU index when using CUDA.
    draft_model
        Optional draft model for speculative decoding (v2 only).  Accepts
        the same size shorthands as ``model_name_or_path`` (a common
        pairing is ``CrisperWhisperModel("large", draft_model="small")``).
    speculative_k
        Speculative-decoding K policy (v2 only).  ``"auto"`` (default)
        self-tunes K to the draft's acceptance with an AIMD +2/-1
        controller whose state persists across chunks, so no hand-tuning is
        needed.  Pass an ``int`` for a fixed K.
    num_speculative_tokens
        Deprecated alias for a fixed ``speculative_k=<int>`` (v2 only).
    min_speculative_tokens, max_speculative_tokens
        Optional explicit adaptive-K window (power users); when ``max > min``
        the controller self-tunes within these bounds instead of the
        ``"auto"`` defaults.
    cache_dir
        Where to store converted CTranslate2 models (v2 only).
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        *,
        backend: str = "auto",
        compute_type: str = "float16",
        device: str = "auto",
        device_index: int = 0,
        draft_model: str | None = None,
        speculative_k: int | str = "auto",
        num_speculative_tokens: int | None = None,
        min_speculative_tokens: int = 0,
        max_speculative_tokens: int = 0,
        cache_dir: str | Path | None = None,
    ):
        from crisperwhisper.version import detect_model_version

        model_name_or_path = resolve_model_id(model_name_or_path)
        if draft_model is not None:
            draft_model = resolve_model_id(draft_model)

        # Resolve the speculative-K policy into (seed, min, max).  Adaptive K
        # is active when max > min; the controller then self-tunes K to the
        # draft's acceptance (AIMD +2/-1, persisted across chunks) and
        # needs no hand-tuned window.  Fixed K uses min == max == 0.
        seed, kmin, kmax = self._resolve_speculative_k(
            speculative_k,
            num_speculative_tokens,
            min_speculative_tokens,
            max_speculative_tokens,
        )
        self._spec_seed = seed
        self._min_speculative_tokens = kmin
        self._max_speculative_tokens = kmax
        num_speculative_tokens = seed

        self._model_path = model_name_or_path
        self._model_version = detect_model_version(model_name_or_path)
        self._backend = self._resolve_backend(backend, self._model_version)

        if self._model_version == 1:
            self._init_v1(
                model_name_or_path, device=device,
                compute_type=compute_type, device_index=device_index,
            )
        elif self._backend == "transformers":
            self._init_v2_transformers(
                model_name_or_path,
                compute_type=compute_type,
                device=device,
                device_index=device_index,
            )
        else:
            self._init_v2(
                model_name_or_path,
                compute_type=compute_type,
                device=device,
                device_index=device_index,
                draft_model=draft_model,
                num_speculative_tokens=num_speculative_tokens,
                cache_dir=cache_dir,
            )

    # Auto self-tuning defaults: a low neutral seed (start conservative and
    # climb only as acceptance warrants) with a wide window the controller
    # never needs to hit.  The equilibrium is set by acceptance, not these.
    _AUTO_SEED = 8
    _AUTO_MIN = 1
    _AUTO_MAX = 16

    @classmethod
    def _resolve_speculative_k(
        cls,
        speculative_k: int | str,
        num_speculative_tokens: int | None,
        min_speculative_tokens: int,
        max_speculative_tokens: int,
    ) -> tuple[int, int, int]:
        """Resolve the speculative-K policy into ``(seed, min, max)``.

        Precedence (most explicit wins):
          1. An explicit adaptive window (``min``/``max`` set) -- power user.
          2. The legacy ``num_speculative_tokens=<int>`` -- fixed K.
          3. ``speculative_k``: ``"auto"`` (default, self-tuning) or an int
             (fixed K).
        """
        if num_speculative_tokens is not None:
            warnings.warn(
                "num_speculative_tokens is deprecated; use "
                "speculative_k=<int> for a fixed K (or leave the default "
                "speculative_k='auto').",
                DeprecationWarning, stacklevel=3,
            )
        if min_speculative_tokens or max_speculative_tokens:
            seed = num_speculative_tokens or cls._AUTO_SEED
            return seed, min_speculative_tokens, max_speculative_tokens
        if num_speculative_tokens is not None:
            return num_speculative_tokens, 0, 0
        if isinstance(speculative_k, str):
            if speculative_k != "auto":
                raise ValueError(
                    f"speculative_k={speculative_k!r}; expected 'auto' or an int."
                )
            return cls._AUTO_SEED, cls._AUTO_MIN, cls._AUTO_MAX
        if isinstance(speculative_k, bool) or not isinstance(speculative_k, int):
            raise ValueError(
                f"speculative_k={speculative_k!r}; expected 'auto' or an int."
            )
        return speculative_k, 0, 0

    @staticmethod
    def _resolve_backend(backend: str, model_version: int) -> str:
        """Resolve ``backend="auto"`` to a concrete backend name."""
        if backend not in ("auto", "ct2", "transformers"):
            raise ValueError(
                f"Unknown backend {backend!r}. Expected 'auto', 'ct2', or "
                "'transformers'."
            )
        if model_version == 1:
            # v1 always uses the legacy HuggingFace pipeline regardless.
            return "transformers"
        if backend != "auto":
            return backend
        if _backend_available("ct2"):
            return "ct2"
        if _backend_available("transformers"):
            return "transformers"
        return "ct2"  # nothing installed: defer to a helpful error at init

    @property
    def backend(self) -> str:
        """The resolved inference backend (``"ct2"`` or ``"transformers"``)."""
        return self._backend

    def _init_v1(
        self,
        model_name_or_path: str,
        *,
        device: str,
        compute_type: str,
        device_index: int = 0,
    ) -> None:
        """Initialize the legacy (v1) model on the Transformers backend.

        The legacy ``nyrahealth/CrisperWhisper`` is a plain Whisper model
        with a changed tokenizer (one explicit space token for sharper word
        timing).  It runs on the same :class:`TransformersEngine` as v2 --
        just with a plain Whisper decoder prefix (no verbatim/intended
        prompt tags) and word timings derived from the explicit space token.
        """
        warnings.warn(_V1_DEPRECATION, DeprecationWarning, stacklevel=3)
        _require_backend("transformers")

        from crisperwhisper.transformers_engine import TransformersEngine

        self._engine = TransformersEngine(
            model_name_or_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
        self._draft_engine = None
        self._num_speculative_tokens = 0

    def _init_v2(
        self,
        model_name_or_path: str,
        *,
        compute_type: str,
        device: str,
        device_index: int,
        draft_model: str | None,
        num_speculative_tokens: int,
        cache_dir: str | Path | None,
    ) -> None:
        """Initialize v2 (CTranslate2) backend."""
        _require_backend("ct2")
        from crisperwhisper.converter import ensure_ct2_model
        from crisperwhisper.engine import CT2Engine

        ct2_path = ensure_ct2_model(
            model_name_or_path,
            quantization=compute_type,
            cache_dir=cache_dir,
        )
        self._engine = CT2Engine(
            ct2_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )

        self._draft_engine: CT2Engine | None = None
        if draft_model is not None:
            draft_ct2_path = ensure_ct2_model(
                draft_model,
                quantization=compute_type,
                cache_dir=cache_dir,
            )
            self._draft_engine = CT2Engine(
                draft_ct2_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
            )

        self._num_speculative_tokens = num_speculative_tokens
        self._spec_decoder = None

    def _get_speculative_decoder(self, mode: str):
        """Return a cached :class:`SpeculativeDecoder`, building it once.

        The decoder wraps the (already loaded) main and draft engines and
        builds a ``TokenRemapper`` over both vocabularies — ~50 ms of work
        that previously ran on *every* ``transcribe`` call.  The engines do
        not change between calls, so we build it lazily once and only update
        the cheap per-call knobs (mode, num_speculative_tokens).
        """
        dec = getattr(self, "_spec_decoder", None)
        if dec is None:
            from crisperwhisper.speculative import SpeculativeDecoder
            dec = SpeculativeDecoder(
                self._engine,
                self._draft_engine,
                num_speculative_tokens=self._num_speculative_tokens,
                mode=mode,  # type: ignore[arg-type]
                min_speculative_tokens=self._min_speculative_tokens,
                max_speculative_tokens=self._max_speculative_tokens,
            )
            self._spec_decoder = dec
        else:
            dec._mode = mode  # type: ignore[assignment]
            dec.num_speculative_tokens = self._num_speculative_tokens
            dec.min_speculative_tokens = self._min_speculative_tokens
            dec.max_speculative_tokens = self._max_speculative_tokens
        return dec

    def _init_v2_transformers(
        self,
        model_name_or_path: str,
        *,
        compute_type: str,
        device: str,
        device_index: int,
    ) -> None:
        """Initialize v2 (HuggingFace Transformers) backend."""
        _require_backend("transformers")
        from crisperwhisper.transformers_engine import TransformersEngine

        self._engine = TransformersEngine(
            model_name_or_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
        self._draft_engine = None
        self._num_speculative_tokens = 0

    @property
    def model_version(self) -> int:
        """Return 1 for legacy CrisperWhisper, 2 for CrisperWhisper 2."""
        return self._model_version

    @staticmethod
    def _validated_longform_config(
        *,
        chunk_duration: float,
        stride: float,
        context_words: int,
        drop_words: int,
        timestamp_aware_drop: bool,
        temperature_fallback: bool,
        max_new_tokens: int,
        early_eot_recovery: bool = True,
    ):
        """Validate the longform window parameters and build a LongformConfig.

        Single validation point shared by :meth:`transcribe` and
        :meth:`transcribe_dual`.
        """
        from crisperwhisper.longform.base import EarlyEotConfig, LongformConfig

        if not 0 < chunk_duration <= 30.0:
            raise ValueError(
                f"chunk_duration must be in (0, 30] seconds (the Whisper "
                f"encoder window); got {chunk_duration}."
            )
        if not 0 < stride <= chunk_duration:
            raise ValueError(
                f"stride must be in (0, chunk_duration={chunk_duration}]; "
                f"got {stride}."
            )
        return LongformConfig(
            chunk_duration=chunk_duration,
            stride=stride,
            context_words=context_words,
            drop_words=drop_words,
            timestamp_aware_drop=timestamp_aware_drop,
            temperature_fallback=temperature_fallback,
            max_new_tokens=max_new_tokens,
            early_eot=EarlyEotConfig(enabled=bool(early_eot_recovery)),
        )

    def _warn_if_hotwords_unsupported(self, hotwords: list[str] | None) -> None:
        """Warn when hotwords are passed to a model without hotword training.

        Hotword boosting is trained into the Pro checkpoints only. The
        standard models were never trained with hotword prompts, so passing
        hotwords can degrade transcription rather than merely doing nothing.
        """
        if not hotwords or self._model_version != 2:
            return
        if "_pro" in str(self._model_path):
            return
        warnings.warn(
            f"hotwords were passed, but '{self._model_path}' does not appear "
            "to be a CrisperWhisper Pro model. Hotword boosting is only "
            "trained into the Pro models; on standard models it is "
            "unsupported and can degrade transcription. Remove the hotwords "
            "argument, or license a Pro model: "
            "https://www.nyra-labs.com/crisperwhisper",
            UserWarning, stacklevel=3,
        )

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str = "en",
        mode: str = "verbatim",
        hotwords: list[str] | None = None,
        sr: int | None = None,
        longform_strategy: str = "continuation",
        chunk_duration: float = 30.0,
        stride: float = 26.0,
        context_words: int = 12,
        drop_words: int = 2,
        timestamp_aware_drop: bool = True,
        temperature_fallback: bool = True,
        max_new_tokens: int = 256,
        speculative_decoding: bool = False,
        speculative_mode: str = "strict",
        hallucination_mitigation: bool = True,
        early_eot_recovery: bool = True,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio.

        Parameters
        ----------
        audio
            File path or numpy array of audio samples.
        language
            ISO 639-1 language code.
        mode
            ``"verbatim"`` or ``"intended"`` (v2 only).
        hotwords
            List of hotword/hint phrases to bias recognition toward.
            Pro models only: standard models were never trained with
            hotword prompts, so passing hotwords can degrade
            transcription and raises a ``UserWarning``.
        sr
            Sample rate when *audio* is a numpy array.
        longform_strategy
            ``"continuation"``, ``"chunked_lcs"``, or ``"token_lcs"``
            (v2 only, audio >30s).
        chunk_duration
            Longform window length in seconds.  Must be ``<= 30`` (the Whisper
            encoder window; features are padded/truncated to 30s).  Default 30.
        stride
            Stride in seconds between longform chunks; the overlap re-covered by
            the next window is ``chunk_duration - stride``.
        context_words
            Number of context words for continuation strategy.
        drop_words
            Cap on how many trailing words may be dropped at a non-final
            continuation chunk boundary.  With ``timestamp_aware_drop`` (default)
            a trailing word is dropped only when its audio starts inside the
            overlap region (so the next window re-covers it); words ending before
            the stride boundary are kept, never lost.
        timestamp_aware_drop
            If ``True`` (default), use per-word timestamps to make the boundary
            drop overlap-aware (computed on every continuation chunk regardless
            of ``word_timestamps``).  If ``False``, use the legacy fixed
            ``drop_words`` count.
        max_new_tokens
            Maximum tokens to generate per chunk.
        speculative_decoding
            Use the draft model for speculative decoding (v2 only).
        speculative_mode
            ``"strict"`` or ``"semantic"`` (v2 only).
        hallucination_mitigation
            Enable post-hoc repetition repair during decoding (v2 only).
            Uses per-ngram-size thresholds from
            ``hallucination.DEFAULT_REPAIR_THRESHOLDS``.
        early_eot_recovery
            Recover a context-conditioned early end-of-text (v2 continuation
            longform only).  Some checkpoints (notably ``large_pro``) can emit an
            over-confident EOT at a sentence-final pause when a continuation
            context is present, truncating the rest of a chunk.  When enabled
            (default) the continuation strategy detects such a premature stop and
            forces the decode past it, guarded so it never hallucinates into
            trailing silence/noise (see
            :class:`~crisperwhisper.longform.base.EarlyEotConfig`).  No-op on
            short audio, and on backends without the required decode primitives.
        word_timestamps
            If ``True`` (v2 only), capture cross-attention during decoding
            and run a Viterbi alignment to populate
            ``TranscriptionResult.words`` with per-word start/end times in
            seconds of the original audio.  On the ``ct2`` backend there is
            no measurable wall-time overhead vs ``word_timestamps=False``
            thanks to bulk on-device attention capture + transfer; the
            ``transformers`` backend recovers attention with one extra
            teacher-forced forward pass.

            Combines with ``speculative_decoding=True`` (ct2): accepted
            draft tokens keep the draft model's cross-attention and
            verifier corrections keep the main model's (a warning notes the
            timings may be slightly less precise).  **Not implemented** with
            a ``longform_strategy`` other than ``"continuation"`` (the
            LCS-stitched strategies need their own per-chunk timing +
            overlap-window merge rule); that combination raises
            :class:`NotImplementedError`.
        alignment_heads
            Explicit ``[(layer, head), ...]`` selection to extract
            attention from.  Defaults to the model's stored alignment heads
            (the CT2 ``config.json``, which the converter copies from the HF
            ``generation_config``; or the HF ``generation_config`` directly
            on the transformers backend).
        suppress_tokens
            Token IDs to suppress during decoding.  ``None`` (default) uses
            the model's ``generation_config.suppress_tokens``; an explicit
            list overrides it for this call, and ``[]`` disables suppression
            entirely.  Applied identically on both backends and on every
            decode path (greedy, repair, speculative, temperature fallback).

        Returns
        -------
        TranscriptionResult
        """
        suppress_tokens = _sanitize_suppress_tokens(suppress_tokens)
        self._warn_if_hotwords_unsupported(hotwords)
        if self._model_version == 1:
            return self._transcribe_v1(
                audio, language=language, mode=mode,
                hotwords=hotwords, sr=sr,
                speculative_decoding=speculative_decoding,
                max_new_tokens=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                alignment_heads=alignment_heads,
                suppress_tokens=suppress_tokens,
            )

        if speculative_decoding and self._backend == "transformers":
            warnings.warn(
                "Speculative decoding is not supported on the transformers "
                "backend and will be ignored. Use backend='ct2' with a "
                "draft_model for speculative decoding.",
                UserWarning, stacklevel=2,
            )
            speculative_decoding = False

        return self._transcribe_v2(
            audio, language=language, mode=mode, hotwords=hotwords, sr=sr,
            longform_strategy=longform_strategy,
            chunk_duration=chunk_duration, stride=stride,
            context_words=context_words, drop_words=drop_words,
            timestamp_aware_drop=timestamp_aware_drop,
            temperature_fallback=temperature_fallback,
            max_new_tokens=max_new_tokens,
            speculative_decoding=speculative_decoding,
            speculative_mode=speculative_mode,
            hallucination_mitigation=hallucination_mitigation,
            early_eot_recovery=early_eot_recovery,
            word_timestamps=word_timestamps,
            alignment_heads=alignment_heads,
            suppress_tokens=suppress_tokens,
        )

    def transcribe_dual(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str = "en",
        modes: tuple[str, ...] = ("verbatim", "intended"),
        hotwords: list[str] | None = None,
        sr: int | None = None,
        longform_strategy: str = "continuation",
        chunk_duration: float = 30.0,
        stride: float = 26.0,
        context_words: int = 12,
        drop_words: int = 2,
        timestamp_aware_drop: bool = True,
        temperature_fallback: bool = True,
        max_new_tokens: int = 256,
        hallucination_mitigation: bool = True,
        early_eot_recovery: bool = True,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[TranscriptionResult, ...]:
        """Transcribe one audio in several modes in a single batched GPU pass.

        ``verbatim`` and ``intended`` differ only by the decoder prompt
        prefix and share the same encoder output, so both are decoded
        together in one batched decoder pass -- the expensive autoregressive
        decode is run once for all modes and the additional transcript(s) are
        nearly free.  Each mode's word-timing cross-attention
        (``word_timestamps=True``) is captured inline during that same pass,
        so there is no extra forward work.

        The decode is greedy with the same token suppression as
        :meth:`transcribe`, so each mode's output matches a per-mode
        :meth:`transcribe` call.  The match is exact up to floating-point
        batched-GEMM rounding: batching two independent rows is mathematically
        identical per row but differs at the ULP level, which can flip a rare
        near-tie token; on long audio such a flip can additionally shift a
        downstream chunk's continuation context.  In practice short audio is
        bit-identical and long audio differs only in rare borderline
        tokens/timings.

        Parameters
        ----------
        audio
            File path or numpy array of audio samples.
        language
            ISO 639-1 language code.
        modes
            Ordered modes to transcribe; each must be ``"verbatim"`` or
            ``"intended"``.  Defaults to ``("verbatim", "intended")``.
        hotwords
            Hotword/hint phrases (shared across modes).
        sr
            Sample rate when *audio* is a numpy array.
        max_new_tokens
            Maximum tokens to generate per mode.
        hallucination_mitigation
            Enable post-hoc repetition repair (per row).
        word_timestamps
            If ``True``, populate ``words`` per mode with word-level timings.
        alignment_heads
            Explicit ``[(layer, head), ...]`` selection for attention.
        suppress_tokens
            Token IDs to suppress during decoding (shared across modes).
            ``None`` uses the model's ``generation_config`` default; ``[]``
            disables suppression.

        longform_strategy
            Longform strategy for audio > 30s.  Only ``"continuation"`` is
            supported for dual-mode.
        stride, context_words, drop_words
            Longform chunking parameters (see :meth:`transcribe`).

        Returns
        -------
        tuple[TranscriptionResult, ...]
            One result per entry in ``modes``, in the same order.

        Notes
        -----
        v2 / ``ct2`` backend only.  Speculative decoding is not combined with
        dual-mode batching yet.  For longform audio *every* chunk's decode is
        shared across modes: when the modes' continuation contexts tokenize to
        different lengths, the native primitive advances the shorter prompt(s)
        with their own real greedy tokens ("catch-up") until the rows line up,
        then decodes them together -- so the batching benefit is not limited
        to equal-length chunks.

        ``early_eot_recovery`` (default on) applies per row: a healthy row is
        dismissed by the cheap gap check, so the shared batched decode is
        unaffected; only a row that actually collapsed pays for a single-prompt
        re-decode of its own recovery.
        """
        self._warn_if_hotwords_unsupported(hotwords)
        if self._model_version == 1:
            raise NotImplementedError(
                "transcribe_dual() requires a CrisperWhisper v2 model "
                "(nyrahealth/CrisperWhisper2)."
            )
        if self._backend != "ct2":
            raise NotImplementedError(
                "transcribe_dual() requires the ct2 backend."
            )

        modes = tuple(modes)
        if not modes:
            raise ValueError("modes must be a non-empty sequence.")
        for m in modes:
            if m not in ("verbatim", "intended"):
                raise ValueError(
                    f"Unknown mode: {m!r}. Expected 'verbatim' or 'intended'."
                )

        from crisperwhisper.prompt import PromptBuilder

        suppress_tokens = _sanitize_suppress_tokens(suppress_tokens)
        lf_config = self._validated_longform_config(
            chunk_duration=chunk_duration,
            stride=stride,
            context_words=context_words,
            drop_words=drop_words,
            timestamp_aware_drop=timestamp_aware_drop,
            temperature_fallback=temperature_fallback,
            max_new_tokens=max_new_tokens,
            early_eot_recovery=early_eot_recovery,
        )

        t0 = time.perf_counter()
        audio_array = load_audio(audio, sr=sr)
        duration = get_duration(audio_array)

        prompt_builder = PromptBuilder(self._engine, language=language)
        hw = [w for w in (hotwords or []) if w.strip()] or None

        if duration <= SHORT_THRESHOLD_S:
            per_mode = self._transcribe_short_dual(
                self._engine, prompt_builder, audio_array,
                modes=modes, hotwords=hw,
                max_new_tokens=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                word_timestamps=word_timestamps,
                alignment_heads=alignment_heads,
                temperature_fallback=temperature_fallback,
                suppress_tokens=suppress_tokens,
            )
            per_mode = [(text, None, words) for text, words in per_mode]
        else:
            if longform_strategy != "continuation":
                raise NotImplementedError(
                    "transcribe_dual() only supports "
                    "longform_strategy='continuation'."
                )
            from crisperwhisper.longform.continuation import (
                continuation_transcribe_dual,
            )

            per_mode = continuation_transcribe_dual(
                self._engine, prompt_builder, audio_array, lf_config,
                modes=modes, hotwords=hw,
                hallucination_mitigation=hallucination_mitigation,
                word_timestamps=word_timestamps,
                alignment_heads=alignment_heads,
                suppress_tokens=suppress_tokens,
            )

        elapsed = time.perf_counter() - t0
        dur = round(duration, 2)
        pt = round(elapsed, 3)
        return tuple(
            TranscriptionResult(
                text=text,
                language=language,
                mode=mode,
                duration=dur,
                processing_time=pt,
                chunks=chunks,
                words=(words or None) if word_timestamps else None,
            )
            for mode, (text, chunks, words) in zip(modes, per_mode)
        )

    def _transcribe_v1(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str,
        mode: str,
        hotwords: list[str] | None,
        sr: int | None,
        speculative_decoding: bool,
        max_new_tokens: int,
        hallucination_mitigation: bool = True,
        alignment_heads: list[tuple[int, int]] | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> TranscriptionResult:
        """Transcribe with a legacy (v1) model on the Transformers backend.

        The legacy model is a plain Whisper model with a changed tokenizer
        (explicit space token), so it only supports verbatim transcription
        with word timestamps -- no intended mode, hotwords, speculative
        decoding, or context-aware longform.  Word timings reuse the same
        Viterbi alignment as v2, but draw the inter-word pause signal from
        the explicit space token's cross-attention (``blank_source="space"``).
        Audio longer than 30s is processed as independent 30s windows.
        """
        from crisperwhisper.result import WordTimestamp

        if mode != "verbatim":
            warnings.warn(
                _V1_FEATURE_WARNING.format(feature="intended mode"),
                UserWarning, stacklevel=3,
            )
        if hotwords:
            warnings.warn(
                _V1_FEATURE_WARNING.format(feature="hotwords"),
                UserWarning, stacklevel=3,
            )
        if speculative_decoding:
            warnings.warn(
                _V1_FEATURE_WARNING.format(feature="speculative decoding"),
                UserWarning, stacklevel=3,
            )

        t0 = time.perf_counter()
        audio_array = load_audio(audio, sr=sr)
        duration = get_duration(audio_array)
        engine = self._engine

        chunk_samples = int(SHORT_THRESHOLD_S * SAMPLE_RATE)
        if duration > SHORT_THRESHOLD_S:
            warnings.warn(
                "The legacy v1 model has no context-aware longform support; "
                "audio > 30s is transcribed as independent 30s windows. "
                "Upgrade to nyrahealth/CrisperWhisper2 for seamless longform.",
                UserWarning, stacklevel=3,
            )

        texts: list[str] = []
        words: list[WordTimestamp] = []
        n = max(1, len(audio_array))
        for start in range(0, n, chunk_samples):
            chunk = audio_array[start:start + chunk_samples]
            if len(chunk) == 0:
                continue
            offset = start / SAMPLE_RATE
            text, chunk_words = self._transcribe_v1_chunk(
                engine, chunk, language=language,
                max_new_tokens=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                alignment_heads=alignment_heads,
                suppress_tokens=suppress_tokens,
            )
            if text:
                texts.append(text)
            for w in chunk_words:
                words.append(WordTimestamp(
                    word=w.word, start=w.start + offset, end=w.end + offset,
                ))

        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=" ".join(texts).strip(),
            language=language,
            mode="verbatim",
            duration=round(duration, 2),
            processing_time=round(elapsed, 3),
            chunks=None,
            words=words or None,
        )

    def _transcribe_v1_chunk(
        self,
        engine,
        audio_chunk: np.ndarray,
        *,
        language: str,
        max_new_tokens: int,
        hallucination_mitigation: bool,
        alignment_heads: list[tuple[int, int]] | None,
        suppress_tokens: list[int] | None = None,
    ):
        """Transcribe a single <=30s window with the legacy model.

        Returns ``(text, words)`` with word timings from the explicit
        space token's cross-attention.
        """
        from crisperwhisper.prompt import strip_prompt_artifacts
        from crisperwhisper.word_timing import extract_word_timings

        prefix = engine.get_decoder_prefix(language)

        engine.enable_attention(alignment_heads)
        features, mel = engine.extract_features_with_mel(audio_chunk)
        gen_ids, attention = engine.generate_with_repair_and_attention(
            features, prefix,
            max_length=max_new_tokens,
            hallucination_mitigation=hallucination_mitigation,
            suppress_tokens=suppress_tokens,
        )
        text = strip_prompt_artifacts(
            engine.decode_tokens(gen_ids, skip_special=True)
        )
        audio_duration_s = min(len(audio_chunk) / SAMPLE_RATE, 30.0)
        words = extract_word_timings(
            engine, gen_ids, attention, mel,
            audio_duration_s=audio_duration_s,
            blank_source="space",
        )
        return text, words

    def _transcribe_v2(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str,
        mode: str,
        hotwords: list[str] | None,
        sr: int | None,
        longform_strategy: str,
        chunk_duration: float = 30.0,
        stride: float,
        context_words: int,
        drop_words: int,
        timestamp_aware_drop: bool = True,
        temperature_fallback: bool = True,
        max_new_tokens: int,
        speculative_decoding: bool,
        speculative_mode: str,
        hallucination_mitigation: bool,
        early_eot_recovery: bool = True,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> TranscriptionResult:
        """Full v2 transcription pipeline (CT2 or transformers backend)."""
        lf_config = self._validated_longform_config(
            chunk_duration=chunk_duration,
            stride=stride,
            context_words=context_words,
            drop_words=drop_words,
            timestamp_aware_drop=timestamp_aware_drop,
            temperature_fallback=temperature_fallback,
            max_new_tokens=max_new_tokens,
            early_eot_recovery=early_eot_recovery,
        )
        from crisperwhisper.longform.chunked_lcs import chunked_lcs_transcribe
        from crisperwhisper.longform.continuation import (
            continuation_transcribe,
            continuation_transcribe_with_word_timestamps,
        )
        from crisperwhisper.longform.token_lcs import token_lcs_transcribe
        from crisperwhisper.prompt import PromptBuilder, strip_prompt_artifacts

        longform_strategies = {
            "continuation": continuation_transcribe,
            "chunked_lcs": chunked_lcs_transcribe,
            "token_lcs": token_lcs_transcribe,
        }

        t0 = time.perf_counter()
        audio_array = load_audio(audio, sr=sr)
        duration = get_duration(audio_array)

        prompt_builder = PromptBuilder(self._engine, language=language)
        hw = [w for w in (hotwords or []) if w.strip()] or None

        engine = self._engine
        use_speculative = (
            speculative_decoding and self._draft_engine is not None
        )

        if word_timestamps and use_speculative:
            # Supported via Option B: accepted *draft* tokens keep the
            # draft model's cross-attention; the always-verified token and
            # verifier *corrections* keep the main model's cross-attention.
            # Because two models' alignment heads are mixed (and the draft
            # model's heads are usually less timing-accurate), warn so the
            # caller knows word timings may be slightly less precise than
            # the non-speculative path.
            warnings.warn(
                "word_timestamps=True with speculative_decoding=True uses "
                "the draft model's cross-attention for accepted draft "
                "tokens and the main model's for verifier corrections. "
                "Word timings may be slightly less accurate than with "
                "speculative_decoding=False. Disable speculative decoding "
                "for the most accurate timings.",
                UserWarning, stacklevel=3,
            )

        if use_speculative:
            engine = self._get_speculative_decoder(speculative_mode)  # type: ignore[assignment]
            # Re-seed the persistent adaptive-K controller for this audio so
            # its first chunk starts from the seed and converges over the
            # following chunks (no effect when K is fixed).
            engine._reset_adaptive_next = True

        words: list = []

        if duration <= SHORT_THRESHOLD_S:
            text, words = self._transcribe_short(
                engine, prompt_builder, audio_array,
                mode=mode, hotwords=hw,
                max_new_tokens=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                word_timestamps=word_timestamps,
                alignment_heads=alignment_heads,
                temperature_fallback=temperature_fallback,
                suppress_tokens=suppress_tokens,
            )
            chunks = None
        else:
            if longform_strategy not in longform_strategies:
                raise ValueError(
                    f"Unknown longform_strategy: {longform_strategy!r}. "
                    f"Expected one of {list(longform_strategies.keys())}."
                )

            if word_timestamps and longform_strategy != "continuation":
                # Not yet implemented: only the continuation strategy
                # has the per-chunk timing + global-seam-monotonisation
                # pass (see ``continuation_transcribe_with_word_timestamps``).
                # Wiring this up for the LCS-stitched strategies is
                # feasible but non-trivial -- the LCS overlap window
                # would need its own attention-row merge rule.  Raise
                # so the caller switches strategies explicitly.
                raise NotImplementedError(
                    f"word_timestamps=True with "
                    f"longform_strategy={longform_strategy!r} is not "
                    f"implemented yet.  Use "
                    f"longform_strategy='continuation' for now, or "
                    f"set word_timestamps=False to use this strategy."
                )

            strategy_fn = longform_strategies[longform_strategy]

            if longform_strategy == "continuation":
                if word_timestamps:
                    text, chunks, words = continuation_transcribe_with_word_timestamps(
                        engine, prompt_builder, audio_array, lf_config,
                        mode=mode, hotwords=hw,
                        hallucination_mitigation=hallucination_mitigation,
                        alignment_heads=alignment_heads,
                        suppress_tokens=suppress_tokens,
                    )
                else:
                    # continuation_transcribe needs alignment_heads too: with
                    # timestamp_aware_drop (default) it captures attention to
                    # make the boundary drop overlap-aware, then discards the
                    # word timestamps.
                    text, chunks = strategy_fn(
                        engine, prompt_builder, audio_array, lf_config,
                        mode=mode, hotwords=hw,
                        hallucination_mitigation=hallucination_mitigation,
                        alignment_heads=alignment_heads,
                        suppress_tokens=suppress_tokens,
                    )
            else:
                text, chunks = strategy_fn(
                    engine, prompt_builder, audio_array, lf_config,
                    mode=mode, hotwords=hw,
                    suppress_tokens=suppress_tokens,
                )

        elapsed = time.perf_counter() - t0

        return TranscriptionResult(
            text=text,
            language=language,
            mode=mode,
            duration=round(duration, 2),
            processing_time=round(elapsed, 3),
            chunks=chunks,
            words=(words or None) if word_timestamps else None,
        )

    def verbatimize(
        self,
        audio: Union[str, Path, np.ndarray],
        transcript: str,
        *,
        language: str = "en",
        sr: int | None = None,
        max_new_tokens: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
    ) -> TranscriptionResult:
        """Transform an intended transcript into verbatim using audio context.

        Only available with CrisperWhisper v2 models.

        Parameters
        ----------
        audio
            File path or numpy array.
        transcript
            The intended (clean) transcript to verbatimize.
        language
            ISO 639-1 language code.
        sr
            Sample rate when *audio* is a numpy array.
        max_new_tokens
            Maximum tokens to generate.
        hallucination_mitigation
            Enable real-time n-gram blocking during decoding.
        suppress_tokens
            Token IDs to suppress; ``None`` uses the model's
            ``generation_config`` default, ``[]`` disables suppression.
        word_timestamps
            If ``True``, capture cross-attention inline during the
            verbatimize decode (same mechanism as :meth:`transcribe`) and
            populate ``result.words`` with per-word start/end seconds.
        alignment_heads
            Explicit ``[(layer, head), ...]`` selection for attention;
            defaults to the model's stored alignment heads.

        Returns
        -------
        TranscriptionResult with ``mode="verbatimize"`` (and ``words`` set
        when ``word_timestamps=True``).
        """
        if self._model_version == 1:
            raise NotImplementedError(
                "verbatimize() requires a CrisperWhisper v2 model "
                "(nyrahealth/CrisperWhisper2). The v1 model does not "
                "support this feature."
            )
        suppress_tokens = _sanitize_suppress_tokens(suppress_tokens)

        from crisperwhisper.prompt import PromptBuilder, strip_prompt_artifacts

        t0 = time.perf_counter()
        audio_array = load_audio(audio, sr=sr)
        duration = get_duration(audio_array)

        if duration > SHORT_THRESHOLD_S:
            logger.warning(
                "Verbatimize is designed for audio <= 30s. "
                "Audio is %.1fs; results may be degraded.", duration,
            )

        prompt_builder = PromptBuilder(self._engine, language=language)
        prompt_tokens = prompt_builder.verbatimize(transcript)

        words: list = []
        if word_timestamps:
            from crisperwhisper.word_timing import extract_word_timings

            self._engine.enable_attention(alignment_heads)
            features, mel = self._engine.extract_features_with_mel(audio_array)
            gen_ids, attention = self._engine.generate_with_repair_and_attention(
                features, prompt_tokens,
                max_length=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                suppress_tokens=suppress_tokens,
            )
            words = extract_word_timings(
                self._engine, gen_ids, attention, mel,
                audio_duration_s=min(duration, 30.0),
            )
        else:
            features = self._engine.extract_features(audio_array)
            gen_ids = self._engine.generate_with_repair(
                features, prompt_tokens,
                max_length=max_new_tokens,
                hallucination_mitigation=hallucination_mitigation,
                suppress_tokens=suppress_tokens,
            )

        text = self._engine.decode_tokens(gen_ids, skip_special=True)
        text = strip_prompt_artifacts(text)

        elapsed = time.perf_counter() - t0

        return TranscriptionResult(
            text=text,
            language=language,
            mode="verbatimize",
            duration=round(duration, 2),
            processing_time=round(elapsed, 3),
            words=(words or None) if word_timestamps else None,
        )

    def forced_align(
        self,
        audio: Union[str, Path, np.ndarray],
        text: str,
        *,
        language: str = "en",
        mode: str = "verbatim",
        sr: int | None = None,
        longform_strategy: str = "continuation",
        hallucination_mitigation: bool = True,
        alignment_heads: list[tuple[int, int]] | None = None,
    ) -> TranscriptionResult:
        """Align a known transcript to the audio (forced alignment).

        Produces word-level timestamps for the *provided* ``text`` using a
        **transcribe-then-align** strategy: the audio is transcribed with the
        normal (longform) pipeline to get the model's hypothesis words *with*
        cross-attention timestamps, then the reference ``text`` is aligned to
        that hypothesis at the word level.  Matched words inherit the hypothesis
        timestamp; unmatched reference words are interpolated between their
        surrounding matched anchors (see :mod:`crisperwhisper.forced_align`).

        This is robust to pauses and long silences (they are represented by the
        hypothesis timestamps) and works on **every backend** (CT2,
        transformers, legacy v1) and any audio length.

        Parameters
        ----------
        audio
            File path or 16 kHz numpy array.
        text
            The transcript to align.  For v2, verbatim-style text aligns best
            (matches the model's verbatim hypothesis most closely).
        language
            ISO 639-1 language code.
        mode
            Transcription mode for the internal hypothesis pass
            (``"verbatim"`` recommended; ignored by the legacy v1 model).
        sr
            Sample rate when *audio* is a numpy array.
        longform_strategy
            Longform strategy for the internal transcription pass; must support
            word timestamps (``"continuation"``).
        hallucination_mitigation
            Enable real-time n-gram blocking during the transcription pass.
        alignment_heads
            Optional explicit ``(layer, head)`` pairs for word timing; defaults
            to the model's stored alignment heads.

        Returns
        -------
        TranscriptionResult with ``mode="forced_align"`` and ``words`` set to
        the aligned :class:`WordTimestamp` list (audio seconds).
        """
        from crisperwhisper.forced_align import align_to_hypothesis

        if not text or not text.strip():
            raise ValueError("forced_align() requires non-empty text.")

        t0 = time.perf_counter()

        hyp = self.transcribe(
            audio,
            language=language,
            mode=mode,
            sr=sr,
            longform_strategy=longform_strategy,
            hallucination_mitigation=hallucination_mitigation,
            word_timestamps=True,
            alignment_heads=alignment_heads,
        )

        words = align_to_hypothesis(text, hyp.words or [], hyp.duration)

        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=" ".join(w.word for w in words),
            language=language,
            mode="forced_align",
            duration=hyp.duration,
            processing_time=round(elapsed, 3),
            chunks=None,
            words=words or None,
        )

    @staticmethod
    def _transcribe_short(
        engine,
        prompt_builder,
        audio: np.ndarray,
        *,
        mode: str,
        hotwords: list[str] | None,
        max_new_tokens: int,
        hallucination_mitigation: bool,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
        temperature_fallback: bool = True,
        suppress_tokens: list[int] | None = None,
    ):
        """Transcribe audio <= 30s (v2 backend).

        Returns ``(text, words)`` where ``words`` is a list of
        :class:`WordTimestamp` if ``word_timestamps=True``, else an empty
        list.  The audio is clipped/padded to the 30s window before
        feature extraction; word timestamps are reported in seconds of
        that window (which equals seconds of the original audio for
        utterances <= 30s).
        """
        from crisperwhisper.audio import SAMPLE_RATE as _SR
        from crisperwhisper.fallback import decode_with_coverage_fallback
        from crisperwhisper.prompt import strip_prompt_artifacts
        from crisperwhisper.word_timing import extract_word_timings

        if mode == "verbatim":
            prompt_tokens = prompt_builder.verbatim(hotwords=hotwords)
            sibling_prompt = prompt_builder.intended(hotwords=hotwords)
        else:
            prompt_tokens = prompt_builder.intended(hotwords=hotwords)
            sibling_prompt = prompt_builder.verbatim(hotwords=hotwords)

        if word_timestamps:
            engine.enable_attention(alignment_heads)
            features, mel = engine.extract_features_with_mel(audio)
            if hasattr(engine, "generate_with_attention"):
                # Speculative decoder: capture draft/main attention per
                # Option B (hallucination repair is applied post-hoc inside).
                gen_ids, attention = engine.generate_with_attention(
                    features, prompt_tokens,
                    max_length=max_new_tokens,
                    hallucination_mitigation=hallucination_mitigation,
                    suppress_tokens=suppress_tokens,
                )
            else:
                gen_ids, attention = engine.generate_with_repair_and_attention(
                    features, prompt_tokens,
                    max_length=max_new_tokens,
                    hallucination_mitigation=hallucination_mitigation,
                    suppress_tokens=suppress_tokens,
                )
            gen_ids, attention = decode_with_coverage_fallback(
                engine, features, mel, prompt_tokens, gen_ids, attention,
                max_length=max_new_tokens, want_attention=True,
                enabled=temperature_fallback, ref_prompt_tokens=sibling_prompt,
                suppress_tokens=suppress_tokens,
            )
            text = engine.decode_tokens(gen_ids, skip_special=True)
            text = strip_prompt_artifacts(text)
            audio_duration_s = min(len(audio) / _SR, 30.0)
            words = extract_word_timings(
                engine, gen_ids, attention, mel,
                audio_duration_s=audio_duration_s,
            )
            return text, words

        features, mel = engine.extract_features_with_mel(audio)
        gen_ids = engine.generate_with_repair(
            features, prompt_tokens,
            max_length=max_new_tokens,
            hallucination_mitigation=hallucination_mitigation,
            suppress_tokens=suppress_tokens,
        )
        gen_ids, _ = decode_with_coverage_fallback(
            engine, features, mel, prompt_tokens, gen_ids, None,
            max_length=max_new_tokens, want_attention=False,
            enabled=temperature_fallback, ref_prompt_tokens=sibling_prompt,
            suppress_tokens=suppress_tokens,
        )

        text = engine.decode_tokens(gen_ids, skip_special=True)
        text = strip_prompt_artifacts(text)
        return text, []

    @staticmethod
    def _transcribe_short_dual(
        engine,
        prompt_builder,
        audio: np.ndarray,
        *,
        modes: tuple[str, ...],
        hotwords: list[str] | None,
        max_new_tokens: int,
        hallucination_mitigation: bool,
        word_timestamps: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
        temperature_fallback: bool = True,
        suppress_tokens: list[int] | None = None,
    ):
        """Batched dual-mode transcription for audio <= 30s (v2 backend).

        Decodes every mode in ``modes`` in a single batched decoder pass
        (the encoder runs once over the repeated mel; each row decodes
        independently), then -- when ``word_timestamps`` is set -- recovers
        each mode's word timings with a teacher-forced attention pass that
        reuses the shared encoder output.

        Returns a list of ``(text, words)`` tuples, one per mode in order;
        ``words`` is empty unless ``word_timestamps=True``.
        """
        from crisperwhisper.audio import SAMPLE_RATE as _SR
        from crisperwhisper.fallback import (
            decode_with_coverage_fallback,
            word_count,
        )
        from crisperwhisper.prompt import strip_prompt_artifacts
        from crisperwhisper.word_timing import extract_word_timings

        prompts = [
            prompt_builder.verbatim(hotwords=hotwords)
            if m == "verbatim"
            else prompt_builder.intended(hotwords=hotwords)
            for m in modes
        ]

        # Shared encoder + one batched decode pass for all modes.  The native
        # primitive encodes the single mel once, decodes every row in lockstep
        # and (when requested) captures each row's word-timing cross-attention
        # inline -- no repeated features, no separate teacher-forced pass.
        features1, mel = engine.extract_features_with_mel(audio)
        if word_timestamps:
            engine.enable_attention(alignment_heads)

        gens, attns = engine.generate_dual_greedy(
            features1, prompts,
            max_length=max_new_tokens,
            hallucination_mitigation=hallucination_mitigation,
            word_timestamps=word_timestamps,
            features_single=features1,
            suppress_tokens=suppress_tokens,
        )

        audio_duration_s = min(len(audio) / _SR, 30.0)

        # Both modes are decoded -- gate each row against its sibling's word
        # count and temperature-recover a collapsed row right away.
        base_counts = [word_count(engine, g) for g in gens]

        out: list[tuple[str, list]] = []
        for i, (gen_ids, mode) in enumerate(zip(gens, modes)):
            attention = attns[i] if (word_timestamps and attns is not None) else None
            ref_n = max(
                (base_counts[k] for k in range(len(gens)) if k != i), default=0,
            )
            gen_ids, attention = decode_with_coverage_fallback(
                engine, features1, mel, prompts[i], gen_ids, attention,
                max_length=max_new_tokens, want_attention=word_timestamps,
                enabled=temperature_fallback, ref_word_count=ref_n,
                suppress_tokens=suppress_tokens,
            )
            text = strip_prompt_artifacts(
                engine.decode_tokens(gen_ids, skip_special=True)
            )
            if word_timestamps and text:
                words = extract_word_timings(
                    engine, gen_ids, attention, mel,
                    audio_duration_s=audio_duration_s,
                )
            else:
                words = []
            out.append((text, words))
        return out
