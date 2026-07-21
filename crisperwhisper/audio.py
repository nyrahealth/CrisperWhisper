"""Audio loading and resampling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

SAMPLE_RATE = 16_000


def load_audio(source: Union[str, Path, np.ndarray], sr: int | None = None) -> np.ndarray:
    """Load audio from a file path or numpy array and resample to 16 kHz.

    Parameters
    ----------
    source
        Path to an audio file (any format supported by soundfile; if librosa
        is installed it is used as a fallback decoder for other formats)
        or a numpy array of audio samples.
    sr
        Sample rate of the input audio when *source* is a numpy array.
        Ignored when *source* is a file path (the sample rate is read from
        the file header).

    Returns
    -------
    Mono float32 numpy array at 16 kHz.
    """
    if isinstance(source, np.ndarray):
        audio = source.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1) if audio.shape[-1] <= audio.shape[0] else audio.mean(axis=0)
        if sr is not None and sr != SAMPLE_RATE:
            audio = _resample(audio, sr, SAMPLE_RATE)
        return audio

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import soundfile as sf
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                f"Could not decode {path} with soundfile ({exc}). "
                "Install librosa for a wider-format fallback decoder: "
                "pip install librosa"
            ) from exc
        audio, file_sr = librosa.load(str(path), sr=None, mono=True)
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if file_sr != SAMPLE_RATE:
        audio = _resample(audio, file_sr, SAMPLE_RATE)

    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    # soxr "HQ" is the same resampler librosa.resample uses by default
    # (res_type="soxr_hq"); calling it directly drops the librosa dependency
    # with bit-identical output.
    import soxr
    return soxr.resample(audio, orig_sr, target_sr, quality="HQ").astype(np.float32)


def get_duration(audio: np.ndarray) -> float:
    """Duration in seconds for a 16 kHz audio array."""
    return len(audio) / SAMPLE_RATE
