"""CTranslate2 Whisper inference engine.

Thin wrapper around ``ctranslate2.models.Whisper`` that exposes a
generate / generate_batch interface aligned with CrisperWhisper's needs
(custom prompt tokens, repetition control, mel feature extraction).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import ctranslate2
import numpy as np
import tokenizers

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128  # Whisper large-v3 uses 128 mel bins
CHUNK_LENGTH_S = 30
CHUNK_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE
N_FRAMES = CHUNK_SAMPLES // HOP_LENGTH  # 3000 frames for 30s

# Fork-only Whisper APIs every pipeline path depends on beyond plain
# ``generate``: hallucination repair and word timing use the incremental /
# bulk attention primitives.  Checked once at engine construction so stock
# ctranslate2 fails fast with a fix-it message instead of an
# ``AttributeError`` mid-transcription.  Feature-specific APIs
# (``generate_dual_greedy`` for transcribe_dual, ``generate_speculative``
# for the native speculative loop) are checked at their entry points via
# :func:`check_fork_feature`; the speculative subset is additionally probed
# by ``crisperwhisper.check_speculative_support``.
_REQUIRED_FORK_APIS = (
    "prefill",
    "forward_step",
    "set_alignment_heads",
    "generate_greedy_with_attention",
)


def _clean_suppress_tokens(tokens) -> list[int]:
    """Drop negative ids (the HF ``-1`` "default set" sentinel).

    CTranslate2 and the numpy argmax paths need explicit token ids; a ``-1``
    would silently suppress the *last* vocab token under numpy indexing.
    """
    return [int(t) for t in tokens if int(t) >= 0]


def _missing_fork_apis_error(missing: list[str]) -> ImportError:
    return ImportError(
        f"The installed ctranslate2 ({ctranslate2.__version__}) is missing "
        f"CrisperWhisper APIs: {', '.join(missing)}.\n"
        "You likely have the upstream package or an outdated fork build. "
        "Fix with:\n"
        "  pip uninstall ctranslate2\n"
        "  pip install --upgrade ctranslate2-crisperwhisper"
    )


def _check_fork_apis(whisper_model) -> None:
    """Fail fast when the installed ctranslate2 lacks the core fork APIs."""
    missing = [m for m in _REQUIRED_FORK_APIS if not hasattr(whisper_model, m)]
    if missing:
        raise _missing_fork_apis_error(missing)


def check_fork_feature(whisper_model, api: str) -> None:
    """Raise a helpful ImportError when a feature-specific fork API is absent.

    Used at feature entry points (e.g. ``generate_dual_greedy`` for
    ``transcribe_dual``) so the core pipeline stays usable on fork builds
    that predate the newest primitives.
    """
    if not hasattr(whisper_model, api):
        raise _missing_fork_apis_error([api])


class CT2Engine:
    """Wraps a CTranslate2 Whisper model for fast inference."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        device_index: int = 0,
        compute_type: str = "default",
        intra_threads: int = 4,
        inter_threads: int = 1,
    ):
        model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

        self.device = device
        self.model_path = model_path
        self.model = ctranslate2.models.Whisper(
            str(model_path),
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )
        _check_fork_apis(self.model)

        self._load_tokenizer(model_path)
        self._load_feature_config(model_path)
        self._init_feature_extractor()

        logger.info(
            "CT2Engine ready: %s on %s (%s)",
            model_path.name, device, compute_type,
        )

    def _load_tokenizer(self, model_path: Path) -> None:
        tok_path = model_path / "tokenizer.json"
        if tok_path.exists():
            self.tokenizer = tokenizers.Tokenizer.from_file(str(tok_path))
        else:
            raise FileNotFoundError(
                f"tokenizer.json not found in {model_path}. "
                "Ensure the model was converted with tokenizer files."
            )
        self._build_special_ids()

    _PROMPT_TAG_PREFIXES = (
        "[verbatim_", "[intended_",
    )

    _V2_MARKER_TOKENS = ("[verbatim_1]", "[intended_1]", "<vtx>", "<htx>")

    def _build_special_ids(self) -> None:
        """Cache frequently used token IDs and detect model version.

        ``all_special_ids`` contains tokens that should be stripped from
        decoded output: Whisper control tokens (``<|...|>``), mode tags
        (``[verbatim_N]``, ``[intended_N]``), and prompt markers
        (``<vtx>``, ``<htx>``, ``<ctx>``, etc.).  Sound-event tokens
        like ``[UH]``, ``[UM]``, ``[cough]`` are intentional output and
        are NOT included.
        """
        vocab = self.tokenizer.get_vocab()
        self.eot_id = vocab.get("<|endoftext|>", None)
        self.sot_id = vocab.get("<|startoftranscript|>", None)
        self.no_timestamps_id = vocab.get("<|notimestamps|>", None)
        self.transcribe_id = vocab.get("<|transcribe|>", None)

        self.model_version = 2 if any(t in vocab for t in self._V2_MARKER_TOKENS) else 1

        self._lang_ids: dict[str, int] = {}
        for token, tid in vocab.items():
            if token.startswith("<|") and token.endswith("|>"):
                lang = token[2:-2]
                if len(lang) == 2 and lang.isalpha():
                    self._lang_ids[lang] = tid

        from crisperwhisper.prompt import PROMPT_MARKER_TOKENS

        self.all_special_ids: set[int] = set()
        for token, tid in vocab.items():
            if token.startswith("<|") and token.endswith("|>"):
                self.all_special_ids.add(tid)
            elif any(token.startswith(p) for p in self._PROMPT_TAG_PREFIXES):
                self.all_special_ids.add(tid)
            elif token in PROMPT_MARKER_TOKENS:
                self.all_special_ids.add(tid)

        logger.info("Detected CrisperWhisper v%d model", self.model_version)

    def _load_feature_config(self, model_path: Path) -> None:
        """Load mel spectrogram and generation config."""
        config_path = model_path / "preprocessor_config.json"
        self.n_mels = N_MELS
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            self.n_mels = cfg.get("feature_size", N_MELS)

        # Normalized once here (negatives / the HF ``-1`` sentinel dropped) so
        # every downstream consumer -- ct2 generate, the numpy argmax loops in
        # hallucination.py / speculative.py, the fork's native decode paths --
        # sees the same explicit id list.
        self.default_suppress_tokens: list[int] = []
        for fname in ("generation_config.json", "config.json"):
            gen_path = model_path / fname
            if gen_path.exists():
                gen_cfg = json.loads(gen_path.read_text())
                if "suppress_tokens" in gen_cfg and isinstance(gen_cfg["suppress_tokens"], list):
                    self.default_suppress_tokens = _clean_suppress_tokens(
                        gen_cfg["suppress_tokens"]
                    )
                    break

        # Read alignment heads from the HF generation_config.json
        # (which the CT2 converter copies into the model dir alongside
        # config.json).  The CT2 binary model.bin also embeds the same
        # list as an attribute consumed by ``align()`` -- we read the
        # JSON here so the Python pipeline can pass it back through
        # ``set_alignment_heads`` for the *_with_attention APIs.  Falls
        # back to config.json for older converted models.
        self.default_alignment_heads: list[tuple[int, int]] | None = None
        for fname in ("generation_config.json", "config.json"):
            cfg_path = model_path / fname
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            raw = cfg.get("alignment_heads")
            if isinstance(raw, list) and raw:
                self.default_alignment_heads = [
                    (int(p[0]), int(p[1])) for p in raw if len(p) == 2
                ]
                break

    def _init_feature_extractor(self) -> None:
        from crisperwhisper.features import FeatureExtractor

        self._feature_extractor = FeatureExtractor(
            feature_size=self.n_mels,
            sampling_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            chunk_length=CHUNK_LENGTH_S,
            n_fft=N_FFT,
        )

    def get_language_id(self, language: str) -> int | None:
        return self._lang_ids.get(language)

    def get_decoder_prefix(self, language: str = "en") -> list[int]:
        """Build the standard Whisper decoder prefix token IDs."""
        prefix: list[int] = []
        if self.sot_id is not None:
            prefix.append(self.sot_id)
        lang_id = self.get_language_id(language)
        if lang_id is not None:
            prefix.append(lang_id)
        if self.transcribe_id is not None:
            prefix.append(self.transcribe_id)
        if self.no_timestamps_id is not None:
            prefix.append(self.no_timestamps_id)
        return prefix

    def encode_text(self, text: str) -> list[int]:
        """Tokenize a text string into token IDs (no special tokens)."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_tokens(self, token_ids: list[int] | np.ndarray, skip_special: bool = True) -> str:
        ids = list(int(t) for t in token_ids)
        if skip_special:
            ids = [t for t in ids if t not in self.all_special_ids]
        return self.tokenizer.decode(ids)

    def extract_features(self, audio: np.ndarray) -> ctranslate2.StorageView:
        """Compute log-mel spectrogram features from raw audio.

        Uses the vendored FeatureExtractor (pure numpy, matches Whisper's
        expected frame count exactly).
        """
        sv, _ = self.extract_features_with_mel(audio)
        return sv

    def extract_features_with_mel(
        self, audio: np.ndarray
    ) -> tuple[ctranslate2.StorageView, np.ndarray]:
        """Like :meth:`extract_features` but also returns the raw mel
        numpy array of shape ``[n_mels, n_mel_frames]``, which downstream
        timing extraction uses to derive per-frame blank probabilities.
        """
        audio = audio.astype(np.float32)
        if len(audio) > CHUNK_SAMPLES:
            audio = audio[:CHUNK_SAMPLES]
        elif len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))

        mel = self._feature_extractor(audio, padding=0)
        features = np.expand_dims(mel, 0)
        return ctranslate2.StorageView.from_array(features), mel

    def extract_features_batch(
        self, audios: list[np.ndarray]
    ) -> list[ctranslate2.StorageView]:
        return [self.extract_features(a) for a in audios]

    def _resolve_suppress(self, suppress_tokens: Sequence[int] | None) -> list[int]:
        """Per-call suppress list (``None`` -> model default), negatives dropped.

        An explicit empty list means "suppress nothing" on every path.
        """
        if suppress_tokens is None:
            return self.default_suppress_tokens
        return _clean_suppress_tokens(suppress_tokens)

    def generate(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
    ) -> list[list[int]]:
        """Generate token IDs for a single audio sample.

        Parameters
        ----------
        features
            Mel spectrogram as a CTranslate2 StorageView.
        prompt_tokens
            List of prompt token ID sequences. Each element corresponds to
            one generation pass (e.g. verbatim and intended in one batch).
        max_length
            Maximum number of **new** tokens to generate (matching HF
            ``max_new_tokens`` semantics).
        beam_size
            Beam width (1 = greedy).
        suppress_tokens
            Token IDs to suppress during generation.

        Returns
        -------
        List of generated token ID sequences (one per prompt).
        """
        # CT2 Whisper's effective generation budget is:
        #   min(ct2_ml // 2, ct2_ml - prompt_len + 1)
        # To guarantee *max_length* new tokens we need:
        #   ct2_ml // 2 >= max_length   AND   ct2_ml - prompt_len + 1 >= max_length
        max_prompt = max(len(p) for p in prompt_tokens) if prompt_tokens else 0
        ct2_max_length = max(max_length * 2, max_length + max_prompt - 1)

        results = self.model.generate(
            features,
            prompt_tokens,
            max_length=ct2_max_length,
            beam_size=beam_size,
            suppress_blank=False,
            suppress_tokens=self._resolve_suppress(suppress_tokens),
            return_scores=False,
            return_no_speech_prob=False,
        )
        return [r.sequences_ids[0] for r in results]

    def generate_sampled(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        temperature: float = 0.8,
        topk: int = 0,
        seed: int = 0,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Single-prompt decode with temperature sampling (coverage fallback).

        ``topk=0`` samples from the full temperature-scaled softmax.  The seed
        is set for reproducibility across runs.
        """
        ctranslate2.set_random_seed(int(seed))
        max_prompt = len(prompt_tokens)
        ct2_max_length = max(max_length * 2, max_length + max_prompt - 1)
        results = self.model.generate(
            features,
            [list(prompt_tokens)],
            max_length=ct2_max_length,
            beam_size=1,
            suppress_blank=False,
            suppress_tokens=self._resolve_suppress(suppress_tokens),
            sampling_temperature=float(temperature),
            sampling_topk=int(topk),
            return_scores=False,
            return_no_speech_prob=False,
        )
        return [int(t) for t in results[0].sequences_ids[0]]

    # ------------------------------------------------------------------
    # Early-EOT recovery primitives (see crisperwhisper.longform.early_eot).
    # Implemented with the fork's incremental prefill/forward_step so we can see
    # per-step logits (P(EOT) + EOT masking) that the batched generate hides.
    # ------------------------------------------------------------------

    def eot_probability(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        gen_ids: list[int],
        *,
        suppress_tokens: list[int] | None = None,
    ) -> float | None:
        """P(EOT) the model assigns at the point where ``gen_ids`` stops.

        Prefills ``prompt_tokens`` then feeds ``gen_ids`` (any trailing EOT
        stripped) through ``forward_batch`` and reads the soft-max probability of
        EOT at the next position -- how confidently the decode chose to stop.
        Suppression is applied so the probability matches the greedy
        distribution.  Returns ``None`` when there is no content or no EOT id.
        """
        if self.eot_id is None:
            return None
        content = list(gen_ids)
        while content and content[-1] == self.eot_id:
            content.pop()
        if not content:
            return None
        sup = self._resolve_suppress(suppress_tokens)
        state, _logits = self.model.prefill(features, list(prompt_tokens))
        batch = self.model.forward_batch(state, list(content))
        arr = np.array(
            batch.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
        )
        logits = arr.reshape(-1, arr.shape[-1])[-1].astype(np.float32)
        if sup:
            logits[sup] = -np.inf
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return float(probs[self.eot_id])

    def greedy_stops_and_decode(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        suppress_tokens: list[int] | None = None,
        min_new_tokens: int = 0,
    ) -> tuple[list[int], float | None]:
        """Greedy decode that reports its stop confidence.

        EOT is suppressed for the first ``min_new_tokens`` generated steps
        (forcing the decode past a premature stop); afterwards greedy runs
        normally.  Returns ``(gen_ids, stop_prob)`` where ``gen_ids`` excludes
        the trailing EOT and ``stop_prob`` is P(EOT) at the step that emitted it
        -- or ``None`` if the decode ran to ``max_length`` without an EOT.
        """
        if max_length <= 0:
            return [], None
        sup = self._resolve_suppress(suppress_tokens)
        eot = self.eot_id
        state, logits = self.model.prefill(features, list(prompt_tokens))
        gen: list[int] = []
        stop_prob: float | None = None
        for step in range(int(max_length)):
            arr = np.array(
                logits.to(ctranslate2.DataType.float32).to_device(
                    ctranslate2.Device.cpu
                )
            )
            lp = arr.reshape(-1, arr.shape[-1])[-1].astype(np.float32)
            if sup:
                lp[sup] = -np.inf
            if eot is not None and step < int(min_new_tokens):
                lp[eot] = -np.inf
            tok = int(lp.argmax())
            if eot is not None and tok == eot:
                shifted = lp - lp.max()
                probs = np.exp(shifted)
                probs /= probs.sum()
                stop_prob = float(probs[eot])
                break
            gen.append(tok)
            logits = self.model.forward_step(state, tok)
        return gen, stop_prob

    def cross_attention_for_tokens(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        gen_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced cross-attention for ``gen_ids`` from raw features.

        Backend-agnostic wrapper around :meth:`attention_for_tokens` (encodes
        ``features`` first) so callers don't manage the encoder StorageView.
        Requires :meth:`enable_attention`.
        """
        encoded = self.model.encode(features)
        return self.attention_for_tokens(encoded, list(prompt_tokens), list(gen_ids))

    def generate_batch(
        self,
        features_list: list[ctranslate2.StorageView],
        prompt_tokens_list: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
    ) -> list[list[int]]:
        """Batch generation across multiple audio samples."""
        all_results: list[list[int]] = []
        for feat, prompt in zip(features_list, prompt_tokens_list):
            result = self.generate(
                feat, [prompt],
                max_length=max_length,
                beam_size=beam_size,
                suppress_tokens=suppress_tokens,
            )
            all_results.extend(result)
        return all_results

    # ------------------------------------------------------------------
    # Repair-aware generation (backend-agnostic entry points).
    #
    # These thin wrappers let the shared orchestration / longform code
    # call ``engine.generate_with_repair(...)`` without importing the
    # CT2-specific functions directly.  The CT2 implementation simply
    # delegates to :mod:`crisperwhisper.hallucination`, so behaviour is
    # byte-for-byte identical to calling those functions; the
    # Transformers backend provides its own native implementation.
    # ------------------------------------------------------------------

    def generate_with_repair(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Greedy decode with optional hallucination repair.

        Returns the generated token IDs (no prompt prefix).
        """
        if not hallucination_mitigation:
            return self.generate(
                features, [prompt_tokens], max_length=max_length,
                suppress_tokens=suppress_tokens,
            )[0]
        from crisperwhisper.hallucination import generate_with_repair

        ids, _ = generate_with_repair(
            self, features, prompt_tokens, max_length=max_length,
            suppress_tokens=suppress_tokens,
        )
        return ids

    def generate_with_repair_and_attention(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Greedy decode with optional repair, also returning the per-token
        cross-attention matrix (``[len(gen_ids), F_enc]``, 1-to-1 with the
        returned tokens).  Requires :meth:`enable_attention` first.
        """
        from crisperwhisper.hallucination import generate_with_repair_and_attention

        ids, attn, _ = generate_with_repair_and_attention(
            self, features, prompt_tokens,
            max_length=max_length,
            max_repairs=3 if hallucination_mitigation else 0,
            suppress_tokens=suppress_tokens,
        )
        return ids, attn

    # ------------------------------------------------------------------
    # Batched dual-mode generation (verbatim + intended in one pass).
    # ------------------------------------------------------------------

    def generate_dual_greedy(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens_list: list[list[int]],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        word_timestamps: bool = False,
        features_single: ctranslate2.StorageView | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[list[int]], list[np.ndarray] | None]:
        """Batched greedy decode of several prompts on one **shared** audio.

        Wraps the native ``generate_dual_greedy`` primitive: the encoder runs
        once and the prompts (which may differ in length, e.g. verbatim vs
        intended with diverging continuation contexts) are decoded together in
        lockstep after a short per-row "catch-up" that equalises their
        lengths.  Output is token-for-token identical to decoding each prompt
        on its own with :meth:`generate_greedy_with_attention` (greedy, same
        suppression), and -- when ``word_timestamps`` is set -- each row's
        per-token cross-attention is captured inline (no extra forward pass).

        ``features`` is the **single-audio** mel StorageView (``[1, n_mels,
        T]``) or pre-encoded features; do not pre-tile it.  When
        ``word_timestamps`` is set, call :meth:`enable_attention` first.

        Returns ``(gens, attns)``: ``gens`` is one token-id list per prompt
        (no prompt prefix; trailing EOT included when that row stopped on
        EOT), and ``attns`` is one ``[len(gen_ids), F_enc]`` float32 matrix
        per prompt (or ``None`` when ``word_timestamps`` is False).
        """
        if self.eot_id is None:
            raise ValueError("engine has no eot_id")
        check_fork_feature(self.model, "generate_dual_greedy")
        supp = self._resolve_suppress(suppress_tokens)

        gens_raw, attns_sv = self.model.generate_dual_greedy(
            features if features_single is None else features_single,
            [list(p) for p in prompt_tokens_list],
            max_new_tokens=int(max_length),
            eot_id=int(self.eot_id),
            suppress_tokens=supp,
            want_attention=word_timestamps,
        )
        gens: list[list[int]] = [list(g) for g in gens_raw]

        attns: list[np.ndarray] | None = None
        if word_timestamps:
            attns = []
            for sv in attns_sv:
                arr = np.array(sv) if sv is not None else None
                if arr is None or arr.size == 0:
                    attns.append(np.zeros((0, 0), dtype=np.float32))
                else:
                    attns.append(arr.astype(np.float32, copy=False))

        if not hallucination_mitigation:
            return gens, attns

        # Per-row repair: only rows that actually loop (rare) are re-decoded
        # singly via the rewind-and-escape path, which needs a batch-1 view.
        from crisperwhisper.hallucination import (
            DEFAULT_REPAIR_THRESHOLDS,
            find_token_loop,
            generate_with_repair,
            generate_with_repair_and_attention,
        )

        feats1 = features_single if features_single is not None else features
        for idx, (prompt, gen) in enumerate(zip(prompt_tokens_list, gens)):
            if find_token_loop(
                gen, min_ngram=1, max_ngram=5, reps=DEFAULT_REPAIR_THRESHOLDS,
            ) is None:
                continue
            if word_timestamps:
                ids, attn, _ = generate_with_repair_and_attention(
                    self, feats1, prompt, max_length=max_length, max_repairs=3,
                    suppress_tokens=supp,
                )
                gens[idx] = ids
                attns[idx] = attn
            else:
                ids, _ = generate_with_repair(
                    self, feats1, prompt, max_length=max_length,
                    suppress_tokens=supp,
                )
                gens[idx] = ids
        return gens, attns

    # ------------------------------------------------------------------
    # Cross-attention extraction (for word-level timings).
    #
    # The CT2 fork supports collecting post-softmax cross-attention rows
    # for a configurable list of (layer, head) pairs on each decode step.
    # The captured rows accumulate inside the WhisperDecoderState and can
    # be bulk-transferred to CPU once decoding is done via
    # :meth:`decode_attention`.
    # ------------------------------------------------------------------

    def enable_attention(
        self,
        heads: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Configure which cross-attention heads to collect.

        Parameters
        ----------
        heads
            Either an explicit list of ``(layer_index, head_index)`` pairs,
            or ``None`` to fall back to the model's saved alignment heads
            (``config.json: alignment_heads``).

        Returns
        -------
        The list of heads actually configured.  Raises ``ValueError`` if
        no heads are available (i.e. ``heads`` is ``None`` and the model
        has none stored).
        """
        if heads is None:
            if not self.default_alignment_heads:
                raise ValueError(
                    "No alignment heads available: pass heads=[...] explicitly "
                    "or convert the model with a generation_config that "
                    "includes alignment_heads."
                )
            heads = list(self.default_alignment_heads)
        else:
            heads = [(int(l), int(h)) for (l, h) in heads]
        self.model.set_alignment_heads(heads)
        self._alignment_heads = heads
        return heads

    def disable_attention(self) -> None:
        """Stop collecting cross-attention on future generation calls."""
        self.model.set_alignment_heads([])
        self._alignment_heads = []

    def prefill_with_attention(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
    ) -> tuple[ctranslate2.WhisperDecoderState, ctranslate2.StorageView, ctranslate2.StorageView]:
        """Encode + prefill, additionally returning the attention row for
        the first generation position.  The row is also appended to
        ``state.collected_attention``.

        Requires :meth:`enable_attention` to have been called first.
        """
        return self.model.prefill_with_attention(features, prompt_tokens)

    def forward_step_with_attention(
        self,
        state: ctranslate2.WhisperDecoderState,
        token_id: int,
    ) -> tuple[ctranslate2.StorageView, ctranslate2.StorageView]:
        """One decoder step that also captures the cross-attention row."""
        return self.model.forward_step_with_attention(state, token_id)

    def forward_step_greedy_with_attention(
        self,
        state: ctranslate2.WhisperDecoderState,
        token_id: int,
    ) -> tuple[int, ctranslate2.StorageView]:
        """Greedy decoder step that also captures cross-attention.
        Argmax stays on the GPU; only the small attention row and the
        picked token id leave the device.
        """
        return self.model.forward_step_greedy_with_attention(state, token_id)

    def forward_batch_with_attention(
        self,
        state: ctranslate2.WhisperDecoderState,
        token_ids: list[int],
    ) -> tuple[ctranslate2.StorageView, ctranslate2.StorageView]:
        """Process a batch of tokens (speculative verification) and also
        return per-position cross-attention.

        Returns ``(logits, attention)`` where ``logits`` is the usual
        ``[1, T, vocab]`` StorageView and ``attention`` is a head-averaged
        ``[T, F_enc]`` float32 CPU array-like StorageView.  ``attention[i]``
        is the row used to predict the token following ``token_ids[i]``.

        Used by speculative decoding's word-timing path to recover
        main-model attention for the always-verified token and verifier
        corrections in the same batched pass.  Requires
        :meth:`enable_attention` to have been called first.
        """
        return self.model.forward_batch_with_attention(state, token_ids)

    def attention_for_tokens(
        self,
        encoded: ctranslate2.StorageView,
        prompt_tokens: list[int],
        gen_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced cross-attention for a **known** token sequence.

        Recovers the per-token cross-attention for ``gen_ids`` without
        re-running the autoregressive decode: one ``prefill_with_attention``
        (the row used to predict ``gen_ids[0]``) followed by one batched
        ``forward_batch_with_attention`` over ``gen_ids[:-1]`` (the rows used
        to predict ``gen_ids[1:]``).  Because the fed tokens are exactly the
        decoded ones and attention is causal, the result is identical to the
        attention an autoregressive decode of the same sequence would have
        captured.

        ``encoded`` is a pre-encoded StorageView (from ``model.encode``);
        passing the same ``encoded`` for several modes shares the encoder.
        Requires :meth:`enable_attention` first.

        Returns a ``[len(gen_ids), F_enc]`` head-averaged, post-softmax
        attention matrix, 1-to-1 with ``gen_ids`` (matching the layout from
        :meth:`generate_with_repair_and_attention`).
        """
        gen_ids = [int(t) for t in gen_ids]
        if not gen_ids:
            return np.zeros((0, 0), dtype=np.float32)

        state, _logits, attn0 = self.model.prefill_with_attention(
            encoded, list(prompt_tokens),
        )
        a0 = np.array(
            attn0.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
        )
        row0 = a0.reshape(-1, a0.shape[-1]).mean(axis=0).astype(np.float32)
        if len(gen_ids) == 1:
            return row0[np.newaxis, :]

        _logits2, batch_attn_sv = self.model.forward_batch_with_attention(
            state, gen_ids[:-1],
        )
        batch_attn = np.array(batch_attn_sv).astype(np.float32)
        return np.vstack([row0[np.newaxis, :], batch_attn])

    def generate_greedy_with_attention(
        self,
        features: ctranslate2.StorageView,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        eot_id: int | None = None,
        suppress_tokens: Sequence[int] | None = None,
        ban_first_tokens: Sequence[int] | None = None,
    ) -> tuple[ctranslate2.WhisperDecoderState, list[int]]:
        """Run a whole greedy decode segment inside one C++ thread-pool job.

        Identical semantics to a ``prefill_with_attention`` followed by a
        Python loop of ``forward_step_greedy_with_attention``, but all
        argmax/suppression/loops stay on the device and there is a
        single Python-to-C++ round-trip for the entire segment.

        ``suppress_tokens`` defaults to the model's stored suppress list.
        ``ban_first_tokens`` is masked only on the very first generated
        step -- used by hallucination-repair to break a loop-starting
        token without affecting later steps.

        Returns ``(state, generated_ids)``; the last id will be
        ``eot_id`` iff generation stopped on EOT.  ``state.collected_attention``
        has length ``len(generated_ids)``.
        """
        if eot_id is None:
            if self.eot_id is None:
                raise ValueError(
                    "engine has no eot_id; pass eot_id=... explicitly"
                )
            eot_id = self.eot_id
        suppress = self._resolve_suppress(suppress_tokens)
        ban = [int(t) for t in (ban_first_tokens or ())]

        return self.model.generate_greedy_with_attention(
            features,
            list(prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            eot_id=int(eot_id),
            suppress_tokens=suppress,
            ban_first_tokens=ban,
        )

    def decode_attention(
        self,
        state: ctranslate2.WhisperDecoderState,
        *,
        average_heads: bool = True,
    ) -> np.ndarray:
        """Pull the accumulated per-step cross-attention to CPU as a
        single ``(num_steps, F_enc)`` float32 matrix (or
        ``(num_steps, num_heads, F_enc)`` when ``average_heads=False``).

        Uses the CT2 fork's ``collected_attention_to_cpu`` which performs
        the concat + (optional) head-mean **on the device** and transfers
        the whole stack to CPU in a **single** PCIe copy.  Returns an
        empty ``(0, 0)`` array if no rows were collected.
        """
        if not len(state.collected_attention):
            return np.zeros((0, 0), dtype=np.float32)
        sv = self.model.collected_attention_to_cpu(state, average_heads=average_heads)
        return np.array(sv)


