"""Vendored FeatureExtractor equivalence tests.

``crisperwhisper.features.FeatureExtractor`` is vendored from faster-whisper
(see the module docstring). If faster-whisper is installed, enforce that the
vendored copy produces bit-identical output; a drift here means faster-whisper
changed its feature pipeline upstream and the vendored copy needs review.
"""

from pathlib import Path

import numpy as np
import pytest

from crisperwhisper.features import FeatureExtractor

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
CHUNK_SAMPLES = 30 * 16_000

faster_whisper = pytest.importorskip(
    "faster_whisper",
    reason="faster-whisper not installed; vendored-copy equivalence not checkable",
)
from faster_whisper.feature_extractor import (  # noqa: E402  (after importorskip)
    FeatureExtractor as UpstreamExtractor,
)


def _inputs():
    rng = np.random.default_rng(0)
    yield "noise", rng.standard_normal(CHUNK_SAMPLES).astype(np.float32) * 0.1
    yield "silence", np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    wav = SAMPLES_DIR / "example_en_1.wav"
    if wav.exists():
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        assert sr == 16_000
        audio = np.pad(audio[:CHUNK_SAMPLES], (0, max(0, CHUNK_SAMPLES - len(audio))))
        yield "speech", audio


@pytest.mark.parametrize("n_mels", [80, 128])
def test_vendored_extractor_matches_upstream(n_mels):
    kwargs = dict(
        feature_size=n_mels,
        sampling_rate=16_000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
    )
    ours = FeatureExtractor(**kwargs)
    theirs = UpstreamExtractor(**kwargs)

    np.testing.assert_array_equal(ours.mel_filters, theirs.mel_filters)

    for name, audio in _inputs():
        ref = theirs(audio, padding=0)
        out = ours(audio, padding=0)
        assert out.shape == (n_mels, 3000), name
        np.testing.assert_array_equal(out, ref, err_msg=f"input={name}")
