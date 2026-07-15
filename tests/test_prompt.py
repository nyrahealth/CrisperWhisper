"""Tests for prompt construction (uses a mock engine)."""

import pytest

from crisperwhisper.prompt import PromptBuilder, strip_prompt_artifacts


class MockEngine:
    """Minimal engine mock for prompt building tests."""

    def __init__(self):
        self._vocab = {}
        self._counter = 100

    def encode_text(self, text: str) -> list[int]:
        tokens = []
        for word in text.replace("]", "] ").replace("[", " [").split():
            if word not in self._vocab:
                self._vocab[word] = self._counter
                self._counter += 1
            tokens.append(self._vocab[word])
        return tokens

    def get_decoder_prefix(self, language: str = "en") -> list[int]:
        return [50258, 50259, 50360]  # sot, en, transcribe (example IDs)


class TestPromptBuilder:
    def setup_method(self):
        self.engine = MockEngine()
        self.builder = PromptBuilder(self.engine, language="en")

    def test_verbatim_basic(self):
        tokens = self.builder.verbatim()
        assert len(tokens) > 0
        # Should end with decoder prefix
        assert tokens[-3:] == [50258, 50259, 50360]

    def test_intended_basic(self):
        tokens = self.builder.intended()
        assert len(tokens) > 0
        assert tokens[-3:] == [50258, 50259, 50360]

    def test_verbatim_vs_intended_differ(self):
        v = self.builder.verbatim()
        i = self.builder.intended()
        # The mode tags are different, so token sequences should differ
        assert v != i

    def test_hotwords_add_tokens(self):
        without = self.builder.verbatim()
        with_hw = self.builder.verbatim(hotwords=["HIPAA", "myocardial"])
        assert len(with_hw) > len(without)

    def test_context_adds_tokens(self):
        without = self.builder.verbatim()
        with_ctx = self.builder.verbatim(context="the last few words")
        assert len(with_ctx) > len(without)

    def test_verbatimize(self):
        tokens = self.builder.verbatimize("this is the intended text")
        assert len(tokens) > 0
        assert tokens[-3:] == [50258, 50259, 50360]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            self.builder._build("invalid_mode")


class TestStripPromptArtifacts:
    def test_strips_verbatim_tags(self):
        text = "[verbatim_1][verbatim_2] hello world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_intended_tags(self):
        text = "[intended_1][intended_2] hello world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_hotword_tags(self):
        text = "hello <htx> HIPAA <ehtx> world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_context_tags(self):
        text = "hello <ctx> some context <ectx> world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_verbatimize_tags(self):
        text = "hello <vtx> some text <evtx> world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_clean_text_unchanged(self):
        text = "hello world this is clean"
        assert strip_prompt_artifacts(text) == text

    def test_normalizes_whitespace(self):
        text = "  hello   world  "
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_sot_eot(self):
        text = "hello <sot> inner <eot> world"
        assert strip_prompt_artifacts(text) == "hello world"

    def test_strips_unclosed_sot(self):
        text = "hello <sot> trailing garbage"
        assert strip_prompt_artifacts(text) == "hello"
