from __future__ import annotations

import json
from dataclasses import dataclass

from crisperwhisper import cli
from crisperwhisper.result import TranscriptionResult, WordTimestamp


@dataclass
class _Call:
    model_args: tuple | None = None
    model_kwargs: dict | None = None
    transcribe_args: tuple | None = None
    transcribe_kwargs: dict | None = None


def _fake_model(call: _Call):
    class FakeModel:
        def __init__(self, *args, **kwargs):
            call.model_args = args
            call.model_kwargs = kwargs

        def transcribe(self, *args, **kwargs):
            call.transcribe_args = args
            call.transcribe_kwargs = kwargs
            return TranscriptionResult(
                text="[um] hello",
                language="en",
                mode="verbatim",
                duration=1.25,
                processing_time=0.5,
                words=[WordTimestamp(word="hello", start=0.2, end=0.8)],
            )

    return FakeModel


def test_transcribe_text_defaults(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "sample.wav"
    audio.touch()
    call = _Call()
    monkeypatch.setattr(cli, "CrisperWhisperModel", _fake_model(call))

    assert cli.main(["transcribe", str(audio)]) == 0

    assert capsys.readouterr().out == "[um] hello\n"
    assert call.model_args == ("small",)
    assert call.model_kwargs == {
        "backend": "transformers",
        "device": "cpu",
        "compute_type": "float32",
    }
    assert call.transcribe_args == (audio,)


def test_transcribe_json_to_file(tmp_path, monkeypatch):
    audio = tmp_path / "sample.wav"
    audio.touch()
    output = tmp_path / "result.json"
    call = _Call()
    monkeypatch.setattr(cli, "CrisperWhisperModel", _fake_model(call))

    code = cli.main(
        [
            "transcribe",
            str(audio),
            "--format",
            "json",
            "--output",
            str(output),
            "--word-timestamps",
            "--mode",
            "intended",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["text"] == "[um] hello"
    assert payload["words"] == [{"word": "hello", "start": 0.2, "end": 0.8}]
    assert call.transcribe_kwargs["word_timestamps"] is True
    assert call.transcribe_kwargs["mode"] == "intended"


def test_missing_audio_is_a_clean_error(tmp_path, capsys):
    missing = tmp_path / "missing.wav"

    assert cli.main(["transcribe", str(missing)]) == 1
    assert "Audio file not found" in capsys.readouterr().err
