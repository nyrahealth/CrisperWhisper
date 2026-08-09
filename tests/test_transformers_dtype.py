from types import SimpleNamespace

import pytest


def test_transformers_loader_uses_dtype_keyword(monkeypatch):
    """Keep the loader compatible with Transformers' current API."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    captured = {}

    class FakeProcessor:
        tokenizer = SimpleNamespace(
            get_vocab=lambda: {
                "<|endoftext|>": 1,
                "<|startoftranscript|>": 2,
                "<|transcribe|>": 3,
                "<|notimestamps|>": 4,
                "<|en|>": 5,
            },
        )
        feature_extractor = SimpleNamespace(feature_size=128)

    class FakeModel:
        config = SimpleNamespace(
            eos_token_id=1,
            decoder_start_token_id=2,
            forced_decoder_ids=None,
            begin_suppress_tokens=None,
            suppress_tokens=None,
            alignment_heads=None,
        )
        generation_config = SimpleNamespace(
            no_timestamps_token_id=4,
            task_to_id={"transcribe": 3},
            lang_to_id={"<|en|>": 5},
            forced_decoder_ids=None,
            begin_suppress_tokens=None,
            suppress_tokens=None,
            alignment_heads=None,
        )

        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        classmethod(lambda _cls, *_args, **_kwargs: FakeProcessor()),
    )

    def load_model(_cls, *_args, **kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoModelForSpeechSeq2Seq,
        "from_pretrained",
        classmethod(load_model),
    )

    from crisperwhisper.transformers_engine import TransformersEngine

    TransformersEngine("unused", device="cpu", compute_type="float32")

    assert "dtype" in captured
    assert "torch_dtype" not in captured
