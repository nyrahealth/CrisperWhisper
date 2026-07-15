"""Tests for audio loading utilities."""

import numpy as np
import pytest

from crisperwhisper.audio import load_audio, get_duration, SAMPLE_RATE


class TestLoadAudio:
    def test_numpy_passthrough_16k(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = load_audio(audio, sr=16000)
        assert result.dtype == np.float32
        assert len(result) == 16000

    def test_numpy_resampling(self):
        audio = np.random.randn(48000).astype(np.float32)  # 1s at 48kHz
        result = load_audio(audio, sr=48000)
        assert result.dtype == np.float32
        # Should be resampled to ~16000 samples
        assert abs(len(result) - 16000) < 100

    def test_stereo_to_mono(self):
        audio = np.random.randn(16000, 2).astype(np.float32)
        result = load_audio(audio, sr=16000)
        assert result.ndim == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/path/audio.wav")


class TestGetDuration:
    def test_one_second(self):
        audio = np.zeros(SAMPLE_RATE)
        assert get_duration(audio) == pytest.approx(1.0)

    def test_thirty_seconds(self):
        audio = np.zeros(30 * SAMPLE_RATE)
        assert get_duration(audio) == pytest.approx(30.0)
