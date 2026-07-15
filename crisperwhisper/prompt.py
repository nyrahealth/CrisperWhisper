"""Prompt construction for CrisperWhisper's custom decoder prompts.

CrisperWhisper was trained with special prompt prefixes that control output
behaviour.  This module constructs the exact token sequences the model expects:

- **Verbatim mode**: ``[verbatim_1][verbatim_2]...[verbatim_5]``
- **Intended mode**: ``[intended_1][intended_2]...[intended_5]``
- **Hotwords**: ``<htx> word1 word2 <ehtx>`` appended after mode tags
- **Continuation context**: ``<ctx> last few words <ectx>`` appended last
- **Verbatimize**: ``<vtx> intended transcript <evtx>`` after verbatim tags
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crisperwhisper.interfaces import EngineProtocol


VERBATIMIZE_START = "<vtx>"
VERBATIMIZE_END = "<evtx>"
HOTWORD_START = "<htx>"
HOTWORD_END = "<ehtx>"
CONTEXT_START = "<ctx>"
CONTEXT_END = "<ectx>"

PROMPT_MARKER_TOKENS = (
    VERBATIMIZE_START, VERBATIMIZE_END,
    HOTWORD_START, HOTWORD_END,
    CONTEXT_START, CONTEXT_END,
)
"""Atomic prompt-marker tokens; stripped from decoded output alongside the
Whisper control tokens and mode tags (see ``all_special_ids`` in the engines)."""


class PromptBuilder:
    """Builds decoder prompt token sequences for CrisperWhisper.

    Parameters
    ----------
    engine
        An engine (either backend) used for tokenization and prefix IDs.
    language
        ISO 639-1 language code (e.g. ``"en"``).
    verbatim_tag_count
        Number of ``[verbatim_N]`` tags to emit.
    intended_tag_count
        Number of ``[intended_N]`` tags to emit.
    """

    def __init__(
        self,
        engine: EngineProtocol,
        language: str = "en",
        verbatim_tag_count: int = 5,
        intended_tag_count: int = 5,
    ):
        self._engine = engine
        self.language = language
        self.verbatim_tag_count = verbatim_tag_count
        self.intended_tag_count = intended_tag_count
        self._decoder_prefix = engine.get_decoder_prefix(language)

    def _mode_tags_text(self, mode: str) -> str:
        if mode == "verbatim":
            return "".join(f"[verbatim_{i}]" for i in range(1, self.verbatim_tag_count + 1))
        elif mode == "intended":
            return "".join(f"[intended_{i}]" for i in range(1, self.intended_tag_count + 1))
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'verbatim' or 'intended'.")

    def _encode(self, text: str) -> list[int]:
        return self._engine.encode_text(text)

    def verbatim(
        self,
        hotwords: list[str] | None = None,
        context: str | None = None,
    ) -> list[int]:
        """Build prompt tokens for verbatim transcription."""
        return self._build("verbatim", hotwords=hotwords, context=context)

    def intended(
        self,
        hotwords: list[str] | None = None,
        context: str | None = None,
    ) -> list[int]:
        """Build prompt tokens for intended (clean) transcription."""
        return self._build("intended", hotwords=hotwords, context=context)

    def verbatimize(self, transcript: str) -> list[int]:
        """Build prompt tokens for the verbatimize task.

        The verbatimize prompt uses verbatim tags followed by the intended
        transcript wrapped in ``<vtx>...<evtx>`` markers.
        """
        text = self._mode_tags_text("verbatim")
        text += f" {VERBATIMIZE_START} {transcript.strip()} {VERBATIMIZE_END}"
        prompt_ids = self._encode(text)
        return prompt_ids + self._decoder_prefix

    def _build(
        self,
        mode: str,
        hotwords: list[str] | None = None,
        context: str | None = None,
    ) -> list[int]:
        text = self._mode_tags_text(mode)
        # Order must match training (data/speech_stew_audio_dataset_with_noise.py):
        # the continuation context is built first ("{mode_tags} <ctx> ... <ectx>")
        # and the hotword suffix is appended *after* it.  Emitting hotwords
        # before context here would not match what the model was trained on.
        if context:
            text += f" {CONTEXT_START} {context} {CONTEXT_END}"
        if hotwords:
            text += f" {HOTWORD_START} {' '.join(hotwords)} {HOTWORD_END}"
        prompt_ids = self._encode(text)
        return prompt_ids + self._decoder_prefix


def strip_prompt_artifacts(text: str) -> str:
    """Remove CrisperWhisper prompt markers from decoded output text."""
    text = re.sub(r"\[verbatim_\d+\]", "", text)
    text = re.sub(r"\[intended_\d+\]", "", text)
    text = re.sub(r"<htx>.*?<ehtx>", "", text, flags=re.DOTALL)
    text = re.sub(r"<vtx>.*?<evtx>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ctx>.*?<ectx>", "", text, flags=re.DOTALL)
    text = re.sub(r"<sot>.*?<eot>", "", text, flags=re.DOTALL)
    if "<sot>" in text:
        text = re.sub(r"<sot>.*", "", text, flags=re.DOTALL)
    return " ".join(text.split()).strip()
