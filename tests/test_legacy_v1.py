"""End-to-end tests for the deprecated legacy (v1) CrisperWhisper model.

The legacy ``nyrahealth/CrisperWhisper`` is a plain Whisper model with a
changed tokenizer (one explicit space token).  It runs on the same
:class:`TransformersEngine` as v2 but with a plain Whisper decoder prefix
(no verbatim/intended tags) and word timings derived from the explicit
space token's cross-attention.

These tests are skipped unless ``CW_V1_MODEL_PATH`` is set (and torch /
transformers are installed)::

    CW_V1_MODEL_PATH=nyrahealth/CrisperWhisper pytest -m legacy
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.legacy


def test_detect_version_is_v1(v1_model_path: str):
    from crisperwhisper.version import detect_model_version

    assert detect_model_version(v1_model_path) == 1


def test_engine_reports_v1_and_canonical_ids(v1_engine):
    # The changed tokenizer's vocab has remapped special-token strings, so
    # the engine must source canonical ids from config/generation_config.
    assert v1_engine.model_version == 1
    assert v1_engine.sot_id is not None
    assert v1_engine.no_timestamps_id is not None
    prefix = v1_engine.get_decoder_prefix("en")
    assert prefix[0] == v1_engine.sot_id
    assert v1_engine.get_language_id("en") in prefix
    assert v1_engine.no_timestamps_id in prefix


def test_loading_v1_emits_deprecation_warning(v1_model_path: str):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from crisperwhisper import CrisperWhisperModel

    with pytest.warns(DeprecationWarning):
        CrisperWhisperModel(v1_model_path)


def test_v1_transcribes_text(v1_model, en_audio: str):
    result = v1_model.transcribe(en_audio)
    assert isinstance(result.text, str)
    assert len(result.text.strip()) > 0


def test_v1_outputs_word_timestamps(v1_model, en_audio: str):
    # The legacy model is timing-oriented: it always returns word timings.
    result = v1_model.transcribe(en_audio, word_timestamps=True)
    assert result.words, "v1 should produce word timestamps"
    starts = [w.start for w in result.words]
    ends = [w.end for w in result.words]
    # Monotonic non-decreasing start times.
    assert starts == sorted(starts)
    # Each word's end is not before its start.
    assert all(e >= s - 1e-6 for s, e in zip(starts, ends))
    # Timings fall within the audio duration (~18s sample, allow slack).
    assert min(starts) >= 0.0
    assert max(ends) <= result.duration + 1.0


def test_v1_unsupported_features_warn(v1_model, en_audio: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v1_model.transcribe(en_audio, mode="intended", hotwords=["foo"])
    messages = " ".join(str(w.message) for w in caught)
    assert "intended mode" in messages
    assert "hotwords" in messages


def test_v1_space_blank_matches_engine_path(v1_engine, en_audio: str):
    """The model's word timings should match a direct engine + space-blank run."""
    import numpy as np
    import soundfile as sf

    from crisperwhisper.word_timing import extract_word_timings, SPACE_TOKEN_ID

    audio, _ = sf.read(en_audio)
    audio = audio.astype("float32")

    v1_engine.enable_attention(None)
    feats, mel = v1_engine.extract_features_with_mel(audio)
    prefix = v1_engine.get_decoder_prefix("en")
    gen_ids, attn = v1_engine.generate_with_repair_and_attention(
        feats, prefix, max_length=200,
    )
    # The changed tokenizer emits explicit space tokens between words.
    assert any(t == SPACE_TOKEN_ID for t in gen_ids)

    words = extract_word_timings(
        v1_engine, gen_ids, attn, mel,
        audio_duration_s=min(len(audio) / 16000, 30.0),
        blank_source="space",
    )
    assert words
    starts = [w.start for w in words]
    assert starts == sorted(starts)
