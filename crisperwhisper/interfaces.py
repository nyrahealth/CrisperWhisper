"""The engine interface shared by both inference backends.

:class:`EngineProtocol` documents the duck-typed surface that
:class:`~crisperwhisper.engine.CT2Engine` and
:class:`~crisperwhisper.transformers_engine.TransformersEngine` both
implement, and that every shared algorithm (prompt building, longform
strategies, word timing, hallucination repair orchestration, coverage
fallback, forced alignment) is written against.  It is a
``typing.Protocol`` -- purely structural, no runtime registration -- so
the engines stay import-independent of each other.

Two optional capabilities are feature-detected with ``hasattr`` rather
than being part of the protocol:

* ``generate_with_attention`` -- only :class:`SpeculativeDecoder`; its
  presence routes the word-timing path through the speculative capture.
* ``generate_dual_greedy`` -- only :class:`CT2Engine`; powers
  ``transcribe_dual``.

``SpeculativeDecoder`` also satisfies the decode subset of this protocol
(it proxies tokenizer/special-id state from its main engine) but does not
implement ``generate_sampled`` / ``cross_attention_for_tokens`` -- which is
why the coverage-based temperature fallback is silently inactive under
speculative decoding.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EngineProtocol(Protocol):
    """Structural type for a CrisperWhisper inference engine."""

    # --- tokenizer / special-id state ---------------------------------
    model_version: int
    eot_id: int | None
    sot_id: int | None
    no_timestamps_id: int | None
    transcribe_id: int | None
    all_special_ids: set[int]
    default_suppress_tokens: list[int]
    default_alignment_heads: list[tuple[int, int]] | None
    n_mels: int

    def get_language_id(self, language: str) -> int | None: ...

    def get_decoder_prefix(self, language: str = "en") -> list[int]: ...

    def encode_text(self, text: str) -> list[int]: ...

    def decode_tokens(
        self, token_ids: Sequence[int] | np.ndarray, skip_special: bool = True,
    ) -> str: ...

    # --- feature extraction -------------------------------------------
    def extract_features(self, audio: np.ndarray): ...

    def extract_features_with_mel(self, audio: np.ndarray): ...

    # --- decoding ------------------------------------------------------
    def generate(
        self,
        features,
        prompt_tokens: list[list[int]],
        *,
        max_length: int = 256,
        beam_size: int = 1,
        suppress_tokens: list[int] | None = None,
    ) -> list[list[int]]: ...

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
    ) -> list[int]: ...

    def generate_with_repair(
        self,
        features,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
    ) -> list[int]: ...

    def generate_with_repair_and_attention(
        self,
        features,
        prompt_tokens: list[int],
        *,
        max_length: int = 256,
        hallucination_mitigation: bool = True,
        suppress_tokens: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray]: ...

    # --- cross-attention (word timing) ---------------------------------
    def enable_attention(
        self, heads: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]: ...

    def disable_attention(self) -> None: ...

    def cross_attention_for_tokens(
        self,
        features,
        prompt_tokens: list[int],
        gen_ids: list[int],
    ) -> np.ndarray: ...
