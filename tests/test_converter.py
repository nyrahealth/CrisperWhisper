"""Unit tests for the HF -> CT2 converter's config handling (converter.py).

Pure-logic tests over the JSON sanitization step: CTranslate2's C++ JSON
parser chokes on nulls/bools, so those keys are stripped -- but the keys
inference depends on (``suppress_tokens``, ``alignment_heads``) must
survive.
"""

from __future__ import annotations

import json

from crisperwhisper.converter import _list_has_null, _sanitize_ct2_configs


class TestListHasNull:
    def test_flat(self):
        assert _list_has_null([1, None, 2]) is True
        assert _list_has_null([1, 2]) is False

    def test_nested_forced_decoder_ids(self):
        # The classic Whisper shape: [[1, null], [2, 50360]]
        assert _list_has_null([[1, None], [2, 50360]]) is True
        assert _list_has_null([[2, 11], [3, 3]]) is False

    def test_empty(self):
        assert _list_has_null([]) is False


class TestSanitizeCT2Configs:
    def _write_config(self, tmp_path, fname, cfg):
        (tmp_path / fname).write_text(json.dumps(cfg))

    def test_nulls_bools_and_null_lists_removed(self, tmp_path):
        self._write_config(tmp_path, "generation_config.json", {
            "forced_decoder_ids": [[1, None], [2, 50360]],
            "is_multilingual": True,
            "return_timestamps": False,
            "prev_sot_token_id": None,
            "suppress_tokens": [1, 2, 220],
            "alignment_heads": [[2, 11], [3, 3]],
            "no_timestamps_token_id": 50364,
        })
        _sanitize_ct2_configs(tmp_path)
        cfg = json.loads((tmp_path / "generation_config.json").read_text())

        assert "forced_decoder_ids" not in cfg      # contains null
        assert "is_multilingual" not in cfg         # bool
        assert "return_timestamps" not in cfg       # bool
        assert "prev_sot_token_id" not in cfg       # null
        # The keys inference relies on survive intact:
        assert cfg["suppress_tokens"] == [1, 2, 220]
        assert cfg["alignment_heads"] == [[2, 11], [3, 3]]
        assert cfg["no_timestamps_token_id"] == 50364

    def test_clean_config_untouched(self, tmp_path):
        original = {"suppress_tokens": [220], "max_length": 448}
        self._write_config(tmp_path, "config.json", original)
        _sanitize_ct2_configs(tmp_path)
        assert json.loads((tmp_path / "config.json").read_text()) == original

    def test_missing_files_ok(self, tmp_path):
        _sanitize_ct2_configs(tmp_path)  # no config files: no-op, no error
