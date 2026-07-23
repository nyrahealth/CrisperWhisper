"""HuggingFace Transformers inference backend for CrisperWhisper v2.

A pure-torch peer to :class:`crisperwhisper.engine.CT2Engine`.  It exposes
the *same* engine surface (feature extraction, prompt-prefixed greedy
generation, hallucination repair, cross-attention capture) so the shared
algorithms -- prompt building, the Viterbi word-timing aligner, the
longform strategies and the rewind/escape hallucination repair -- run on
top of it unchanged.

Differences from the CTranslate2 backend:

* No speculative decoding (the draft/verify attention stitching depends on
  the CT2 fork's KV primitives).
* Cross-attention for word timing is captured **inline during generation**
  (``generate(output_attentions=True, return_dict_in_generate=True)``) in a
  single pass -- no separate teacher-forced re-run -- to mirror the CT2
  backend's on-the-fly capture.  This requires the model to be loaded with
  eager attention (``attn_implementation="eager"``); SDPA / flash attention
  do not return attention weights.  The teacher-forced single pass
  (:meth:`TransformersEngine._cross_attention_rows`) is retained only as the
  *forced-aligner* primitive (known token sequence in, attention out).
* Slower than CTranslate2 (no fused kernels / int8).

Requires the ``crisperwhisper[transformers]`` extra (``transformers`` +
``torch``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
CHUNK_LENGTH_S = 30
CHUNK_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE


def _resolve_torch_dtype(compute_type: str):
    """Map a CrisperWhisper ``compute_type`` string onto a torch dtype."""
    import torch

    if compute_type in ("float16", "int8_float16"):
        return torch.float16
    if compute_type == "bfloat16":
        return torch.bfloat16
    if compute_type in ("float32", "int8", "default"):
        return torch.float32
    return torch.float16


class _FirstStepBan:
    """Logits processor that bans tokens only on the first generated step.

    Mirrors the CT2 hallucination-repair behaviour where the loop-starting
    token is masked for a single escape step (and only that step), so the
    model is nudged out of the loop without being forbidden the token
    forever.
    """

    def __init__(self, prefix_len: int, banned: list[int]):
        self.prefix_len = int(prefix_len)
        self.banned = [int(t) for t in banned]

    def __call__(self, input_ids, scores):  # noqa: D401 - HF processor proto
        if self.banned and input_ids.shape[1] == self.prefix_len:
            scores[:, self.banned] = float("-inf")
        return scores


class _EotGate:
    """Logits processor for early-EOT recovery (see ``longform.early_eot``).

    Bans EOT for the first ``min_new_tokens`` generated steps -- forcing the
    greedy decode past a premature stop -- and records ``stop_prob``, the
    soft-max probability of EOT at the step the decode stops on.  Suppression is
    re-applied here so ``stop_prob`` is measured over the same masked
    distribution greedy decoding argmaxes over.
    """

    def __init__(self, prefix_len, eot_id, suppress_ids, min_new_tokens):
        self.prefix_len = int(prefix_len)
        self.eot_id = eot_id
        self.suppress = [int(t) for t in (suppress_ids or [])]
        self.min_new_tokens = int(min_new_tokens)
        self.stop_prob = None

    def __call__(self, input_ids, scores):  # noqa: D401 - HF processor proto
        import torch

        step = input_ids.shape[1] - self.prefix_len
        if self.suppress:
            scores[:, self.suppress] = float("-inf")
        if self.eot_id is not None and step < self.min_new_tokens:
            scores[:, self.eot_id] = float("-inf")
        if self.eot_id is not None and int(torch.argmax(scores[0])) == self.eot_id:
            self.stop_prob = float(
                torch.softmax(scores[0].float(), dim=-1)[self.eot_id]
            )
        return scores


class TransformersEngine:
    """Wraps a HuggingFace Whisper model for CrisperWhisper inference."""

    _PROMPT_TAG_PREFIXES = ("[verbatim_", "[intended_")
    _V2_MARKER_TOKENS = ("[verbatim_1]", "[intended_1]", "<vtx>", "<htx>")

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str = "auto",
        device_index: int = 0,
        compute_type: str = "float16",
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = f"cuda:{device_index}"

        self.device = torch.device(device)
        self.torch_dtype = _resolve_torch_dtype(compute_type)
        self.model_path = str(model_name_or_path)

        logger.info(
            "Loading Transformers model %s on %s (%s)...",
            model_name_or_path, device, self.torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="eager",  # required for output_attentions
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        # We always force the exact decoder prompt ourselves via
        # ``decoder_input_ids``; clear any model-default forced ids so the
        # generate() override does not prepend its own language/task tokens.
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None
        # Match the CT2 backend's first-token behaviour (it decodes with
        # ``suppress_blank=False`` and never applies begin_suppress_tokens):
        # leaving HF's begin-suppression active would forbid EOT as the first
        # token and skew the first-step distribution between backends.
        self.model.generation_config.begin_suppress_tokens = None
        self.model.config.begin_suppress_tokens = None

        self.tokenizer = self.processor.tokenizer
        # 128-mel fallback matches CT2Engine's N_MELS (all v2 checkpoints are
        # v3-family, 128 bins); the processor config normally provides it.
        self.n_mels = int(getattr(self.processor.feature_extractor, "feature_size", 128))

        self._build_special_ids()
        self._load_generation_defaults()
        self._alignment_heads: list[tuple[int, int]] | None = None

        logger.info(
            "TransformersEngine ready: %s (v%d, %d mel bins) on %s",
            Path(self.model_path).name, self.model_version, self.n_mels, device,
        )

    # ------------------------------------------------------------------
    # Tokenizer / special-id bookkeeping (mirrors CT2Engine).
    # ------------------------------------------------------------------

    def _build_special_ids(self) -> None:
        vocab = self.tokenizer.get_vocab()
        cfg = self.model.config
        gc = self.model.generation_config

        # The canonical special-token IDs come from the model config /
        # generation_config rather than the tokenizer vocab.  This matters for
        # the legacy CrisperWhisper "changed tokenizer", whose vocab contains
        # remapped/duplicate special-token strings (e.g. "<|startoftranscript|>"
        # maps to a phantom id) while the model actually uses the standard
        # Whisper ids.  For v2 models the two agree, so behaviour is unchanged.
        self.eot_id = getattr(cfg, "eos_token_id", None) or vocab.get("<|endoftext|>")
        self.sot_id = (
            getattr(cfg, "decoder_start_token_id", None)
            or vocab.get("<|startoftranscript|>")
        )
        self.no_timestamps_id = (
            getattr(gc, "no_timestamps_token_id", None)
            or vocab.get("<|notimestamps|>")
        )

        task_to_id = getattr(gc, "task_to_id", None) or {}
        self.transcribe_id = (
            int(task_to_id["transcribe"]) if "transcribe" in task_to_id
            else vocab.get("<|transcribe|>")
        )

        self.model_version = (
            2 if any(t in vocab for t in self._V2_MARKER_TOKENS) else 1
        )

        # Language ids: prefer generation_config.lang_to_id (authoritative for
        # the changed tokenizer), fall back to scanning the vocab.
        self._lang_ids: dict[str, int] = {}
        lang_to_id = getattr(gc, "lang_to_id", None)
        if lang_to_id:
            for token, tid in lang_to_id.items():
                lang = token[2:-2]
                if len(lang) == 2 and lang.isalpha():
                    self._lang_ids[lang] = int(tid)
        else:
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
        # Include canonical specials that may be absent/remapped in the vocab.
        for tid in (self.eot_id, self.sot_id, self.no_timestamps_id,
                    self.transcribe_id, *self._lang_ids.values()):
            if tid is not None:
                self.all_special_ids.add(int(tid))

    def _load_generation_defaults(self) -> None:
        gc = self.model.generation_config
        cfg = self.model.config

        raw_sup = getattr(gc, "suppress_tokens", None)
        if raw_sup is None:
            raw_sup = getattr(cfg, "suppress_tokens", None)
        # Negatives (the HF ``-1`` "default set" sentinel) are dropped so both
        # backends expose the same explicit id list.
        self.default_suppress_tokens: list[int] = (
            [int(t) for t in raw_sup if int(t) >= 0] if raw_sup else []
        )

        raw_heads = getattr(gc, "alignment_heads", None)
        if raw_heads is None:
            raw_heads = getattr(cfg, "alignment_heads", None)
        self.default_alignment_heads: list[tuple[int, int]] | None = (
            [(int(p[0]), int(p[1])) for p in raw_heads if len(p) == 2]
            if raw_heads else None
        )

    def _resolve_suppress(self, suppress_tokens: list[int] | None) -> list[int]:
        """Per-call suppress list (``None`` -> model default), negatives dropped.

        An explicit empty list means "suppress nothing", matching the CT2
        backend's semantics.
        """
        if suppress_tokens is None:
            return self.default_suppress_tokens
        return [int(t) for t in suppress_tokens if int(t) >= 0]

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
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(
        self, token_ids: list[int] | np.ndarray, skip_special: bool = True
    ) -> str:
        ids = [int(t) for t in token_ids]
        if skip_special:
            ids = [t for t in ids if t not in self.all_special_ids]
        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Feature extraction.
    # ------------------------------------------------------------------

    def extract_features(self, audio: np.ndarray):
        return self.extract_features_with_mel(audio)[0]

    def extract_features_with_mel(self, audio: np.ndarray):
        """Return ``(input_features_tensor, mel_np[n_mels, n_frames])``.

        The HuggingFace ``WhisperFeatureExtractor`` produces a log-mel
        spectrogram at a 10ms hop (= twice the 20ms encoder frame rate),
        which is exactly what :func:`word_timing.blank_logp_from_mel_energy`
        expects.  The numpy mel is returned for blank-energy estimation;
        the tensor (on device, model dtype) feeds generation.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) > CHUNK_SAMPLES:
            audio = audio[:CHUNK_SAMPLES]
        feats = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        )
        inp = feats.input_features  # [1, n_mels, n_frames]
        mel = inp[0].detach().cpu().numpy().astype(np.float32)
        inp = inp.to(self.device, self.torch_dtype)
        return inp, mel

    def extract_features_batch(self, audios: list[np.ndarray]) -> list:
        return [self.extract_features(a) for a in audios]

    # ------------------------------------------------------------------
    # Generation.
    # ------------------------------------------------------------------

    def _run_generate(
        self,
        features,
        prefix: list[int],
        max_new: int,
        *,
        ban_first: set[int] | None = None,
        suppress_tokens: list[int] | None = None,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[int]:
        """Greedy/beam/sampled decode forcing ``prefix`` as the decoder prompt.

        Returns the generated token IDs only (the prompt prefix is
        stripped; HuggingFace's Whisper ``generate`` already omits it, but
        we strip defensively in case a future version includes it).
        """
        import torch
        from transformers import LogitsProcessorList

        if max_new <= 0:
            return []

        dec = torch.tensor([list(prefix)], device=self.device, dtype=torch.long)
        # Always pass the resolved list (matching the CT2 backend): an explicit
        # ``[]`` must *disable* suppression rather than fall back to the
        # model's generation_config default.
        sup = self._resolve_suppress(suppress_tokens)

        kwargs: dict = dict(
            decoder_input_ids=dec,
            max_new_tokens=int(max_new),
            num_beams=int(num_beams),
            do_sample=bool(do_sample),
            suppress_tokens=list(sup),
        )
        if do_sample:
            kwargs["temperature"] = float(temperature)
            kwargs["top_k"] = int(top_k)  # 0 disables top-k filtering in HF
        if ban_first:
            procs = LogitsProcessorList()
            procs.append(_FirstStepBan(len(prefix), list(ban_first)))
            kwargs["logits_processor"] = procs

        with torch.no_grad():
            out = self.model.generate(features, **kwargs)

        seq = [int(t) for t in out[0].tolist()]
        if seq[: len(prefix)] == list(prefix):
            seq = seq[len(prefix):]
        return seq

    def _resolved_alignment_heads(self) -> list[tuple[int, int]]:
        heads = self._alignment_heads
        if heads is None:
            heads = self.default_alignment_heads
        if not heads:
            raise ValueError(
                "No alignment heads configured; call enable_attention(...) "
                "first."
            )
        return heads

    def _stack_step_attention(
        self, cross_attentions, heads: list[tuple[int, int]], n_gen: int,
    ) -> np.ndarray:
        """Assemble per-step cross-attention into a ``[n_gen, F_enc]`` matrix.

        ``cross_attentions`` is the tuple HuggingFace ``generate`` returns when
        ``output_attentions=True``: one entry per generated token, each a tuple
        over decoder layers of ``[1, n_heads, q_len, F_enc]``.  Step 0's
        ``q_len`` is the prompt length (the prefill), later steps' is 1; in both
        cases the **last** query row is the one that predicts that step's token,
        so row ``k`` of the result is the attention used to predict
        ``gen_ids[k]`` -- the same 1-to-1 convention as the CT2 backend.
        """
        import torch

        if not cross_attentions or n_gen <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        F_enc = cross_attentions[0][0].shape[-1]
        rows = []
        for step in cross_attentions:
            acc = torch.zeros((F_enc,), dtype=torch.float32, device=step[0].device)
            for (layer, head) in heads:
                acc += step[layer][0, head, -1].to(torch.float32)
            acc /= max(len(heads), 1)
            rows.append(acc)
        mat = torch.stack(rows, dim=0)[:n_gen]
        return mat.detach().cpu().numpy().astype(np.float32)

    def _run_generate_with_attention(
        self,
        features,
        prefix: list[int],
        max_new: int,
        *,
        ban_first: set[int] | None = None,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Greedy decode forcing ``prefix``, capturing cross-attention inline.

        Single forward-generation pass (no teacher-forced re-run): returns
        ``(gen_ids, attention)`` where ``attention`` is the head-averaged
        ``[len(gen_ids), F_enc]`` matrix, 1-to-1 with ``gen_ids`` (row ``k``
        predicts ``gen_ids[k]``).  Requires :meth:`enable_attention` first.
        """
        import torch
        from transformers import LogitsProcessorList

        heads = self._resolved_alignment_heads()
        if max_new <= 0:
            return [], np.zeros((0, 0), dtype=np.float32)

        dec = torch.tensor([list(prefix)], device=self.device, dtype=torch.long)
        sup = self._resolve_suppress(suppress_tokens)

        kwargs: dict = dict(
            decoder_input_ids=dec,
            max_new_tokens=int(max_new),
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
            use_cache=True,
            suppress_tokens=list(sup),
        )
        if ban_first:
            procs = LogitsProcessorList()
            procs.append(_FirstStepBan(len(prefix), list(ban_first)))
            kwargs["logits_processor"] = procs

        with torch.no_grad():
            out = self.model.generate(features, **kwargs)

        seq = [int(t) for t in out.sequences[0].tolist()]
        gen_ids = seq[len(prefix):] if seq[: len(prefix)] == list(prefix) else seq
        attention = self._stack_step_attention(
            out.cross_attentions, heads, len(gen_ids),
        )
        return gen_ids, attention

    def generate(
        self,
        features,
        prompt_tokens: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
    ) -> list[list[int]]:
        """Generate token IDs for each prompt in ``prompt_tokens``."""
        return [
            self._run_generate(
                features, p, max_length,
                suppress_tokens=suppress_tokens, num_beams=beam_size,
            )
            for p in prompt_tokens
        ]

    def generate_sampled(
        self,
        features,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        temperature: float = 0.8,
        topk: int = 0,
        seed: int = 0,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Single-prompt decode with temperature sampling (coverage fallback).

        ``topk=0`` disables top-k filtering (sample from the full
        temperature-scaled softmax).  Seeded for reproducibility.
        """
        import torch

        torch.manual_seed(int(seed))
        return self._run_generate(
            features, list(prompt_tokens), max_length,
            suppress_tokens=suppress_tokens, do_sample=True,
            temperature=float(temperature), top_k=int(topk),
        )

    # ------------------------------------------------------------------
    # Early-EOT recovery primitives (see crisperwhisper.longform.early_eot).
    # ------------------------------------------------------------------

    def eot_probability(
        self,
        features,
        prompt_tokens: list[int],
        gen_ids: list[int],
        *,
        suppress_tokens: list[int] | None = None,
    ) -> float | None:
        """P(EOT) the model assigns at the point where ``gen_ids`` stops.

        Teacher-forces ``prompt_tokens + gen_ids`` (any trailing EOT stripped) in
        one forward pass and returns the soft-max probability of the end-of-text
        token at the next position -- how confidently the decode chose to stop.
        Suppression is applied to the logits so the probability is measured over
        the same masked distribution the greedy decode saw.  Returns ``None``
        when there is no content to score or the engine has no EOT id.
        """
        import torch

        if self.eot_id is None:
            return None
        content = list(gen_ids)
        while content and content[-1] == self.eot_id:
            content.pop()
        if not content:
            return None
        sup = self._resolve_suppress(suppress_tokens)
        dec = torch.tensor(
            [list(prompt_tokens) + content], device=self.device, dtype=torch.long
        )
        with torch.no_grad():
            out = self.model(
                input_features=features, decoder_input_ids=dec, use_cache=False,
            )
        logits = out.logits[0, -1].float()
        if sup:
            logits[torch.tensor(sup, device=logits.device, dtype=torch.long)] = float(
                "-inf"
            )
        return float(torch.softmax(logits, dim=-1)[self.eot_id])

    def greedy_stops_and_decode(
        self,
        features,
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
        -- or ``None`` if the decode ran to ``max_length`` without an EOT (i.e.
        it did not reach a clean stop, which the recovery gate treats as
        non-confident).
        """
        import torch
        from transformers import LogitsProcessorList

        if max_length <= 0:
            return [], None
        dec = torch.tensor(
            [list(prompt_tokens)], device=self.device, dtype=torch.long
        )
        sup = self._resolve_suppress(suppress_tokens)
        gate = _EotGate(
            prefix_len=len(prompt_tokens), eot_id=self.eot_id,
            suppress_ids=sup, min_new_tokens=int(min_new_tokens),
        )
        with torch.no_grad():
            out = self.model.generate(
                features, decoder_input_ids=dec, max_new_tokens=int(max_length),
                num_beams=1, do_sample=False, suppress_tokens=list(sup),
                logits_processor=LogitsProcessorList([gate]),
            )
        seq = [int(t) for t in out[0].tolist()]
        gen = (
            seq[len(prompt_tokens):]
            if seq[: len(prompt_tokens)] == list(prompt_tokens)
            else seq
        )
        # HF's Whisper ``generate`` does not append the EOT token to the returned
        # sequence, so we rely on the gate -- ``gate.stop_prob`` is set iff EOT
        # became the argmax (the greedy stop) and is ``None`` iff the decode ran
        # to ``max_length`` without stopping.  Strip a trailing EOT defensively.
        if gen and self.eot_id is not None and gen[-1] == self.eot_id:
            gen = gen[:-1]
        return gen, gate.stop_prob

    def cross_attention_for_tokens(
        self,
        features,
        prompt_tokens: list[int],
        gen_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced cross-attention for ``gen_ids`` (backend-agnostic name).

        Thin wrapper over :meth:`_cross_attention_rows`; mirrors the CT2
        engine's method so the fallback path is backend-agnostic.
        """
        return self._cross_attention_rows(features, list(prompt_tokens), list(gen_ids))

    def generate_batch(
        self,
        features_list: list,
        prompt_tokens_list: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
    ) -> list[list[int]]:
        out: list[list[int]] = []
        for feat, prompt in zip(features_list, prompt_tokens_list):
            out.extend(self.generate(
                feat, [prompt],
                max_length=max_length, beam_size=beam_size,
                suppress_tokens=suppress_tokens,
            ))
        return out

    # ------------------------------------------------------------------
    # Cross-attention configuration + capture.
    # ------------------------------------------------------------------

    def enable_attention(
        self, heads: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        """Select which ``(layer, head)`` cross-attention pairs to use for
        word timing.  ``None`` falls back to the model's stored
        ``generation_config.alignment_heads``.
        """
        if heads is None:
            if not self.default_alignment_heads:
                raise ValueError(
                    "No alignment heads available: pass heads=[...] explicitly "
                    "or use a model whose generation_config includes "
                    "alignment_heads."
                )
            heads = list(self.default_alignment_heads)
        else:
            heads = [(int(l), int(h)) for (l, h) in heads]
        self._alignment_heads = heads
        return heads

    def disable_attention(self) -> None:
        self._alignment_heads = []

    def _cross_attention_rows(
        self, features, prompt_tokens: list[int], gen_ids: list[int],
    ) -> np.ndarray:
        """Teacher-forced forward pass recovering the per-token cross
        attention as a head-averaged ``[len(gen_ids), F_enc]`` matrix.

        Row ``k`` is the (post-softmax) cross-attention computed at the
        decoder position that *predicts* ``gen_ids[k]`` -- i.e. while
        consuming the token at absolute position ``len(prompt)+k-1``.  This
        matches the 1-to-1 row/token convention the CT2 backend produces.

        This is the **forced-aligner** primitive: it teacher-forces a *known*
        full token sequence and recovers its attention in one pass.  It is
        *not* used by the transcription/timing path -- that path captures
        attention inline during generation (see
        :meth:`generate_with_repair_and_attention` /
        :meth:`_run_generate_with_attention`), so transcription stays a single
        pass.  Kept for an explicit forced-alignment caller.
        """
        import torch

        if not gen_ids:
            return np.zeros((0, 0), dtype=np.float32)

        heads = self._alignment_heads
        if heads is None:
            heads = self.default_alignment_heads
        if not heads:
            raise ValueError(
                "No alignment heads configured; call enable_attention(...) "
                "first."
            )

        full = list(prompt_tokens) + list(gen_ids)
        dec = torch.tensor([full], device=self.device, dtype=torch.long)
        with torch.no_grad():
            out = self.model(
                input_features=features,
                decoder_input_ids=dec,
                output_attentions=True,
                use_cache=False,
            )
        cross = out.cross_attentions  # tuple[L] of [1, H, L_full, F_enc]

        T = len(gen_ids)
        start = len(prompt_tokens) - 1
        F_enc = cross[0].shape[-1]
        acc = torch.zeros((T, F_enc), dtype=torch.float32, device=self.device)
        for (layer, head) in heads:
            a = cross[layer][0, head]  # [L_full, F_enc]
            acc += a[start:start + T].to(torch.float32)
        acc /= max(len(heads), 1)
        return acc.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Repair-aware generation (backend-agnostic entry points).
    # ------------------------------------------------------------------

    def generate_with_repair(
        self,
        features,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        detect_reps: int | dict[int, int] | None = None,
        keep_reps: int = 1,
        max_ngram: int = 5,
        max_repairs: int = 3,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]:
        """Greedy decode with optional rewind/escape hallucination repair.

        Native re-implementation of the CT2 ``generate_with_repair``
        control flow (free decode -> detect consecutive n-gram loop ->
        rewind to ``keep_reps`` copies -> force one escape step banning the
        loop starter -> continue), reusing the pure loop-detection helpers
        from :mod:`crisperwhisper.hallucination`.
        """
        if not hallucination_mitigation:
            return self._run_generate(
                features, prompt_tokens, max_length,
                suppress_tokens=suppress_tokens,
            )

        from crisperwhisper.hallucination import (
            DEFAULT_REPAIR_THRESHOLDS,
            find_token_loop,
        )

        if detect_reps is None:
            detect_reps = DEFAULT_REPAIR_THRESHOLDS

        generated = self._run_generate(
            features, prompt_tokens, max_length,
            suppress_tokens=suppress_tokens,
        )

        for attempt in range(1, max_repairs + 1):
            hit = find_token_loop(
                generated, min_ngram=1, max_ngram=max_ngram, reps=detect_reps,
            )
            if hit is None:
                break

            loop_start, gram = hit
            n = len(gram)
            keep_end = loop_start + n * keep_reps
            trimmed = generated[:keep_end]
            remaining = max_length - len(trimmed)

            reps_for_log = (
                detect_reps.get(n, "?")
                if isinstance(detect_reps, dict) else detect_reps
            )
            logger.info(
                "Repair %d: %d-gram loop at pos %d (%s reps), "
                "rewinding to %d tokens, banning token %d",
                attempt, n, loop_start, reps_for_log, keep_end, gram[0],
            )

            if remaining <= 0:
                generated = trimmed
                break

            tail = self._run_generate(
                features, list(prompt_tokens) + trimmed, remaining,
                ban_first={gram[0]},
                suppress_tokens=suppress_tokens,
            )
            generated = trimmed + tail

        return generated

    def generate_with_repair_and_attention(
        self,
        features,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        detect_reps: int | dict[int, int] | None = None,
        keep_reps: int = 1,
        max_ngram: int = 5,
        max_repairs: int = 3,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        """Greedy decode with optional repair, capturing cross-attention
        **inline during generation** (single pass -- no teacher-forced re-run),
        mirroring the CT2 backend's on-the-fly capture.

        The returned attention is 1-to-1 with ``gen_ids`` (row ``k`` predicts
        token ``k``).  On a detected n-gram loop the kept-prefix attention rows
        are retained and the continuation is re-decoded with the loop starter
        banned, again capturing attention inline, then concatenated -- so the
        common zero-repair case is exactly one generation pass.

        Requires :meth:`enable_attention` (or stored alignment heads).
        """
        if not hallucination_mitigation:
            return self._run_generate_with_attention(
                features, prompt_tokens, max_length,
                suppress_tokens=suppress_tokens,
            )

        from crisperwhisper.hallucination import (
            DEFAULT_REPAIR_THRESHOLDS,
            find_token_loop,
        )

        if detect_reps is None:
            detect_reps = DEFAULT_REPAIR_THRESHOLDS

        generated, attn = self._run_generate_with_attention(
            features, prompt_tokens, max_length,
            suppress_tokens=suppress_tokens,
        )

        for attempt in range(1, max_repairs + 1):
            hit = find_token_loop(
                generated, min_ngram=1, max_ngram=max_ngram, reps=detect_reps,
            )
            if hit is None:
                break

            loop_start, gram = hit
            n = len(gram)
            keep_end = loop_start + n * keep_reps
            trimmed = generated[:keep_end]
            kept_attn = attn[:keep_end]
            remaining = max_length - len(trimmed)

            reps_for_log = (
                detect_reps.get(n, "?")
                if isinstance(detect_reps, dict) else detect_reps
            )
            logger.info(
                "Repair %d: %d-gram loop at pos %d (%s reps), "
                "rewinding to %d tokens, banning token %d",
                attempt, n, loop_start, reps_for_log, keep_end, gram[0],
            )

            if remaining <= 0:
                generated, attn = trimmed, kept_attn
                break

            tail_ids, tail_attn = self._run_generate_with_attention(
                features, list(prompt_tokens) + trimmed, remaining,
                ban_first={gram[0]},
                suppress_tokens=suppress_tokens,
            )
            generated = trimmed + tail_ids
            if kept_attn.size == 0:
                attn = tail_attn
            elif tail_attn.size == 0:
                attn = kept_attn
            else:
                attn = np.concatenate([kept_attn, tail_attn], axis=0)

        return generated, attn
