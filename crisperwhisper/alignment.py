"""Word timing with the nyra forced aligner.

CrisperWhisper times words with one of two methods:

- ``"attention"`` (default): cross-attention + Viterbi over the model's own
  decode (:mod:`crisperwhisper.word_timing`).  Every language, no extra model.
- ``"forced_aligner"``: the transcript is force-aligned to the audio with
  `nyra-forced-aligner <https://github.com/nyrahealth/nyra-forced-aligner>`_
  (WavLM features + HMM decode).  Much tighter word boundaries, plus
  per-word ``timing_confidence``, ``kind`` (word / filler / sound / cutoff)
  and phone boundaries.  Languages: ``nyra_align.SUPPORTED_LANGUAGES``
  (English only for now); other languages warn and keep the attention
  timings.

Either way ``TranscriptionResult.words`` is the same list of
:class:`~crisperwhisper.result.WordTimestamp` -- the forced aligner only
replaces ``start``/``end`` and fills the extra fields.  The aligner models
load from the Hugging Face Hub (``nyralabs/nyra_forced_aligner_<lang>`` and
``..._pro``) or from a local directory.

``nyra_align`` (and with it torch) is imported lazily, on first use.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from crisperwhisper.audio import SAMPLE_RATE, get_duration, load_audio
from crisperwhisper.result import (
    PhoneTimestamp,
    TranscriptionResult,
    WordTimestamp,
)

TIMING_METHODS = ("attention", "forced_aligner")

_KINDS = {"w": "word", "f": "filler", "s": "sound", "c": "cutoff"}


def validate_timing_method(timing_method: str) -> None:
    if timing_method not in TIMING_METHODS:
        raise ValueError(
            f"Unknown timing_method: {timing_method!r}. "
            f"Expected one of {list(TIMING_METHODS)}."
        )


def _norm_lang(language: str) -> str:
    return language.strip().lower().replace("_", "-").split("-")[0]


class ForcedAligner:
    """Forced alignment of a transcript to audio (nyra-forced-aligner).

    Usable on its own -- no ASR model needed::

        from crisperwhisper import ForcedAligner
        result = ForcedAligner().align("audio.wav", "so i [UM] went there")

    or as the word-timing backend of
    :class:`~crisperwhisper.CrisperWhisperModel` via
    ``timing_method="forced_aligner"`` (pass an instance as the model's
    ``forced_aligner`` argument to configure or share it).

    Parameters
    ----------
    pro
        Use the noise-robust aligner model (``nyralabs/nyra_forced_aligner_<lang>_pro``).
    model
        Explicit aligner model: a Hugging Face repo id or local directory.
        Overrides the per-language default (and ``pro``); the aligner is then
        only used for the language that model was trained for.
    device
        ``"cuda"``, ``"cpu"`` or ``None`` (auto: CUDA when available).
    precision
        ``"fp16"``, ``"bf16"`` or ``"fp32"``; ``None`` = fp16 on CUDA, fp32
        on CPU.
    segment_gap
        ``None`` (default) aligns the whole file in one pass.  A number
        splits the transcript at attention pauses longer than that many
        seconds when re-timing transcribed words
        (``timing_method="forced_aligner"``) and aligns each segment against
        its own stretch of audio (see :meth:`refine`): faster on long audio
        with long pauses, but no more accurate.
    segment_pad
        Seconds of audio around each segment's first and last word.

    The aligner model for a language is downloaded and loaded on first use
    and cached on the instance.
    """

    def __init__(
        self,
        *,
        pro: bool = False,
        model: Union[str, Path, None] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        segment_gap: Optional[float] = None,
        segment_pad: float = 0.5,
    ):
        self.pro = pro
        self.model = model
        self.device = device
        self.precision = precision
        self.segment_gap = segment_gap
        self.segment_pad = segment_pad
        self._aligners: dict = {}

    @staticmethod
    def supported_languages() -> tuple[str, ...]:
        """Languages with a published aligner model."""
        return tuple(_import_nyra().SUPPORTED_LANGUAGES)

    def supports(self, language: str) -> bool:
        return _norm_lang(language) in self.supported_languages()

    def _aligner(self, language: str):
        lang = _norm_lang(language)
        aligner = self._aligners.get(lang)
        if aligner is None:
            nyra = _import_nyra()
            aligner = nyra.Aligner(
                self.model, language=lang, pro=self.pro,
                device=self.device, precision=self.precision,
            )
            self._aligners[lang] = aligner
        return aligner

    def align(
        self,
        audio: Union[str, Path, np.ndarray],
        text: str,
        *,
        language: str = "en",
        sr: int | None = None,
    ) -> TranscriptionResult:
        """Align ``text`` to ``audio``; any length.

        Returns a :class:`TranscriptionResult` with ``mode="forced_align"``,
        ``timing_method="forced_aligner"`` and one word per aligned
        whitespace token of ``text``.  Tokens the aligner has no
        pronunciation for (pure punctuation, unpronounceable symbols) are
        left out of ``words``.

        Raises ``ValueError`` for an unsupported language and
        ``RuntimeError`` when the text cannot be aligned to the audio at all.
        """
        if not text or not text.strip():
            raise ValueError("align() requires non-empty text.")
        if not self.supports(language):
            raise ValueError(
                f"The forced aligner does not support language {language!r} "
                f"yet; supported: {list(self.supported_languages())}."
            )
        t0 = time.perf_counter()
        audio_array = load_audio(audio, sr=sr)
        aligned = self._aligner(language).align(audio_array, text)
        words = [_to_word_timestamp(w) for w in aligned.words]
        return TranscriptionResult(
            text=" ".join(w.word for w in words),
            language=language,
            mode="forced_align",
            duration=round(get_duration(audio_array), 2),
            processing_time=round(time.perf_counter() - t0, 3),
            words=words or None,
            timing_method="forced_aligner",
        )

    def refine(
        self,
        words: List[WordTimestamp],
        audio_array: np.ndarray,
        *,
        language: str,
    ) -> List[WordTimestamp]:
        """Re-time ``words`` (16 kHz ``audio_array``) with the forced aligner.

        The word list is kept 1-to-1: each word is force-aligned as written
        and gets the aligner's ``start``/``end``, ``timing_confidence``,
        ``kind`` and ``phones``; all other fields (``probability``,
        ``tokens``) are kept.  A word the aligner has no pronunciation for
        keeps its attention timing (clamped between its re-timed
        neighbours).

        With ``segment_gap`` set, the words' attention timings split the
        transcript wherever attention sees a pause longer than
        ``segment_gap`` seconds, and each segment is aligned only against
        its own stretch of audio (``segment_pad`` seconds either side).
        A segment that fails to align keeps its attention timings (with a
        warning); ``RuntimeError`` is raised only when every segment fails
        (always the case for a failed one-pass alignment).
        """
        segments = self._segments(words, len(audio_array) / SAMPLE_RATE)
        by_word: dict[int, list] = {}
        failed = 0
        for lo, hi, t0, t1 in segments:
            crop = audio_array[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)]
            try:
                parts = self._align_segment(words[lo:hi], crop, language)
            except RuntimeError:
                failed += 1
                continue
            for i, ps in parts.items():
                by_word[lo + i] = [_shift(a, t0) for a in ps]
        if segments and failed == len(segments):
            raise RuntimeError("no segment of the transcript could be aligned")
        if failed:
            warnings.warn(
                f"Forced alignment failed for {failed} of {len(segments)} "
                "transcript segments; their words keep attention timings.",
                UserWarning, stacklevel=2,
            )

        out: list[WordTimestamp] = []
        for i, w in enumerate(words):
            parts = by_word.get(i)
            if not parts:
                out.append(w)
                continue
            confs = [a.confidence for a in parts if a.confidence is not None]
            out.append(replace(
                w,
                start=parts[0].start,
                end=parts[-1].end,
                timing_confidence=min(confs) if confs else None,
                kind=_KINDS.get(parts[0].type, parts[0].type),
                phones=[PhoneTimestamp(ph.phone, ph.start, ph.end)
                        for a in parts for ph in a.phones],
            ))
        _clamp_unaligned(out, aligned_idx=set(by_word))
        return out

    def _segments(
        self, words: List[WordTimestamp], duration: float,
    ) -> list[tuple[int, int, float, float]]:
        """``(lo, hi, t0, t1)`` per segment: ``words[lo:hi]`` aligned
        against audio ``[t0, t1)`` seconds.  Segments break at attention
        pauses > ``segment_gap``; a segment's audio reaches ``segment_pad``
        beyond its first/last word, but never past the middle of the pause
        to its neighbour."""
        bounds = []
        lo = 0
        for i in range(1, len(words) + 1):
            if i == len(words) or (
                self.segment_gap is not None
                and words[i].start - words[i - 1].end > self.segment_gap
            ):
                bounds.append((lo, i))
                lo = i
        out = []
        for k, (lo, hi) in enumerate(bounds):
            t0 = max(0.0, words[lo].start - self.segment_pad)
            t1 = min(duration, words[hi - 1].end + self.segment_pad)
            if k > 0:
                t0 = max(t0, (words[lo - 1].end + words[lo].start) / 2)
            if k + 1 < len(bounds):
                t1 = min(t1, (words[hi - 1].end + words[hi].start) / 2)
            out.append((lo, hi, t0, t1))
        return out

    def _align_segment(
        self, words: List[WordTimestamp], audio_array: np.ndarray, language: str,
    ) -> dict[int, list]:
        """Align ``words`` to ``audio_array``; maps word index -> the
        aligner's words for it (times relative to ``audio_array``)."""
        # One aligner token per whitespace piece of each word.  The aligner
        # returns its words in transcript order with ``word`` = the raw
        # token, and drops only tokens it cannot pronounce -- a decision that
        # depends on the token string alone, so a dropped token never equals
        # an aligned one and an in-order string match maps back exactly.
        pieces: list[str] = []
        owner: list[int] = []
        for i, w in enumerate(words):
            for piece in w.word.split():
                pieces.append(piece)
                owner.append(i)
        if not pieces:
            return {}

        aligned = self._aligner(language).align(audio_array, " ".join(pieces))

        by_word: dict[int, list] = {}
        p = 0
        for aw in aligned.words:
            while p < len(pieces) and pieces[p] != aw.word:
                p += 1
            if p == len(pieces):
                raise RuntimeError(
                    f"aligned word {aw.word!r} does not match the transcript"
                )
            by_word.setdefault(owner[p], []).append(aw)
            p += 1
        return by_word


def _shift(aligned_word, offset: float):
    """Copy of a nyra ``Word`` with all times moved by ``offset`` seconds."""
    return replace(
        aligned_word,
        start=aligned_word.start + offset,
        end=aligned_word.end + offset,
        phones=[replace(p, start=p.start + offset, end=p.end + offset)
                for p in aligned_word.phones],
    )


def _clamp_unaligned(words: List[WordTimestamp], aligned_idx: set) -> None:
    """In-place: keep attention-timed words inside the gap between their
    force-aligned neighbours, so the timeline stays forward-going without
    moving any force-aligned word."""
    for i, w in enumerate(words):
        if i in aligned_idx:
            continue
        lo = words[i - 1].end if i > 0 else 0.0
        hi = next((words[j].start for j in range(i + 1, len(words))
                   if j in aligned_idx), None)
        start, end = max(w.start, lo), max(w.end, lo)
        if hi is not None and hi >= lo:
            start, end = min(start, hi), min(end, hi)
        if (start, end) != (w.start, w.end):
            words[i] = replace(w, start=start, end=end)


def _to_word_timestamp(w) -> WordTimestamp:
    return WordTimestamp(
        word=w.word,
        start=w.start,
        end=w.end,
        timing_confidence=w.confidence,
        kind=_KINDS.get(w.type, w.type),
        phones=[PhoneTimestamp(p.phone, p.start, p.end) for p in w.phones],
    )


def apply_timing_method(
    result: TranscriptionResult,
    audio_array: np.ndarray,
    timing_method: str,
    get_aligner,
) -> TranscriptionResult:
    """Finalize ``result.words`` for ``timing_method`` (in place).

    ``get_aligner`` is a zero-argument callable returning the
    :class:`ForcedAligner` (so it is only built when actually used).
    Falls back to the attention timings with a ``UserWarning`` when the
    language is not supported by the forced aligner or the alignment fails.
    """
    if timing_method == "forced_aligner":
        aligner = get_aligner()
        if not aligner.supports(result.language):
            warnings.warn(
                f"timing_method='forced_aligner' supports only "
                f"{list(aligner.supported_languages())} so far; using "
                f"attention word timings for language {result.language!r}.",
                UserWarning, stacklevel=4,
            )
            timing_method = "attention"
    if result.words is None:
        return result
    result.timing_method = "attention"
    if timing_method != "forced_aligner":
        return result
    try:
        result.words = aligner.refine(
            result.words, audio_array, language=result.language,
        )
    except RuntimeError as exc:
        warnings.warn(
            f"Forced alignment failed ({exc}); using attention word timings.",
            UserWarning, stacklevel=4,
        )
        return result
    result.timing_method = "forced_aligner"
    return result


def _import_nyra():
    try:
        import nyra_align
    except ImportError as exc:
        raise ImportError(
            "timing_method='forced_aligner' needs nyra-forced-aligner, which "
            "is part of the crisperwhisper install. Reinstall with:\n"
            "  pip install --upgrade crisperwhisper"
        ) from exc
    return nyra_align


__all__ = [
    "TIMING_METHODS",
    "ForcedAligner",
    "apply_timing_method",
    "validate_timing_method",
]
