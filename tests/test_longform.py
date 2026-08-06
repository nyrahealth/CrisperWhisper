"""Tests for longform stitching logic (pure-logic, no model needed)."""

import numpy as np
import pytest

from crisperwhisper.longform.base import LongformConfig, make_chunks
from crisperwhisper.longform.chunked_lcs import _longest_common_subsequence
from crisperwhisper.longform.continuation import _overlap_drop_index
from crisperwhisper.longform.token_lcs import _find_longest_common_token_sequence
from crisperwhisper.result import WordTimestamp

SAMPLE_RATE = 16_000


def _wt(start, end, word="w"):
    """Build a WordTimestamp (start/end may be None for unplaceable words)."""
    return WordTimestamp(word=word, start=start, end=end)


class TestMakeChunks:
    def test_short_audio_single_chunk(self):
        audio = np.zeros(10 * SAMPLE_RATE)  # 10 seconds
        config = LongformConfig()
        chunks = make_chunks(audio, config)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10 * SAMPLE_RATE

    def test_exactly_30s(self):
        audio = np.zeros(30 * SAMPLE_RATE)
        config = LongformConfig()
        chunks = make_chunks(audio, config)
        assert len(chunks) == 1

    def test_60s_audio(self):
        audio = np.zeros(60 * SAMPLE_RATE)
        config = LongformConfig(chunk_duration=30.0, stride=26.0)
        chunks = make_chunks(audio, config)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 30 * SAMPLE_RATE

    def test_overlap_exists(self):
        audio = np.zeros(60 * SAMPLE_RATE)
        config = LongformConfig(chunk_duration=30.0, stride=26.0)
        chunks = make_chunks(audio, config)
        total_samples = sum(len(c) for c in chunks)
        assert total_samples > len(audio)  # overlap means more total samples

    def test_custom_stride(self):
        audio = np.zeros(60 * SAMPLE_RATE)
        config = LongformConfig(chunk_duration=30.0, stride=15.0)
        chunks = make_chunks(audio, config)
        # With stride=15 and 60s audio, we expect more chunks than stride=26
        assert len(chunks) >= 3


class TestOverlapDrop:
    """The overlap-aware boundary drop for continuation longform.

    ``_overlap_drop_index`` returns the index up to which words are confirmed
    (kept); ``words[idx:]`` are dropped and re-transcribed by the next window.
    With chunk_duration=30 and stride=26 the overlap region is [26s, 30s].
    """

    def test_word_before_overlap_kept(self):
        # Regression for the reported bug: speech ends well before the stride
        # boundary -> the trailing word must NOT be dropped (the next window,
        # starting at 26s, would never re-cover it).
        word_ts = [_wt(0.0, 1.0), _wt(1.0, 2.0), _wt(19.0, 20.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 3  # nothing dropped

    def test_word_in_overlap_dropped(self):
        word_ts = [_wt(0.0, 1.0), _wt(24.0, 25.0), _wt(27.0, 28.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 2  # the 27s word (in [26,30]) is dropped

    def test_drop_words_caps_overlap_drop(self):
        # Three words start inside the overlap, but the cap limits the drop.
        word_ts = [
            _wt(0.0, 1.0), _wt(10.0, 11.0),
            _wt(26.5, 27.0), _wt(27.5, 28.0), _wt(28.5, 29.0),
        ]
        keep1 = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep1 == 4  # only the last word dropped (cap=1)
        keep3 = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=3,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep3 == 2  # all three overlap words dropped

    def test_drop_words_never_crosses_overlap(self):
        # A large cap must still never drop a word that starts before overlap.
        word_ts = [_wt(0.0, 1.0), _wt(10.0, 11.0), _wt(27.0, 28.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=10,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 2  # only the single overlap word is eligible

    def test_configurable_chunk_and_stride(self):
        # Non-default stride shifts the overlap region.
        word_ts = [_wt(0.0, 1.0), _wt(14.0, 15.0), _wt(16.0, 17.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=15.0, drop_words=2,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 2  # 14s word kept, 16s word (>=15) dropped

    def test_straddling_word_kept(self):
        # A word starting before the stride boundary is kept even if it ends
        # inside the overlap.
        word_ts = [_wt(0.0, 1.0), _wt(25.0, 27.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 2

    def test_final_chunk_keeps_all(self):
        word_ts = [_wt(0.0, 1.0), _wt(27.0, 28.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=True,
        )
        assert keep == 2

    def test_no_timings_falls_back_to_drop_words(self):
        word_ts = [_wt(None, None), _wt(None, None), _wt(None, None)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 2  # legacy fixed-count drop of the last word

    def test_toggle_off_uses_legacy(self):
        # Even with a word in the overlap, the toggle-off path drops the fixed
        # count regardless of timing.
        word_ts = [_wt(0.0, 1.0), _wt(19.0, 20.0)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=False, is_last=False,
        )
        assert keep == 1  # legacy drops the last word blindly

    def test_unplaceable_trailing_word(self):
        # A trailing unplaceable word doesn't move the boundary by itself and
        # is kept (we can't prove it's in the overlap, so don't lose it).
        word_ts = [_wt(0.0, 1.0), _wt(10.0, 11.0), _wt(None, None)]
        keep = _overlap_drop_index(
            word_ts, stride_sec=26.0, drop_words=1,
            timestamp_aware_drop=True, is_last=False,
        )
        assert keep == 3


class TestLongestCommonSubsequence:
    def test_identical_sequences(self):
        seq = ["hello", "world", "foo"]
        sa, sb, length = _longest_common_subsequence(seq, seq)
        assert length == 3

    def test_partial_overlap(self):
        a = ["the", "quick", "brown", "fox"]
        b = ["brown", "fox", "jumps", "over"]
        sa, sb, length = _longest_common_subsequence(a, b)
        assert length == 2
        assert a[sa:sa + length] == ["brown", "fox"]

    def test_no_overlap(self):
        a = ["hello", "world"]
        b = ["foo", "bar"]
        _, _, length = _longest_common_subsequence(a, b)
        assert length == 0

    def test_case_insensitive(self):
        a = ["Hello", "World"]
        b = ["hello", "world"]
        _, _, length = _longest_common_subsequence(a, b)
        assert length == 2

    def test_empty_sequences(self):
        _, _, length = _longest_common_subsequence([], ["a", "b"])
        assert length == 0
        _, _, length = _longest_common_subsequence(["a"], [])
        assert length == 0


class TestTokenLCS:
    def test_no_overlap(self):
        seqs = [[1, 2, 3], [4, 5, 6]]
        merged = _find_longest_common_token_sequence(seqs, set())
        assert merged == [1, 2, 3, 4, 5, 6]

    def test_perfect_overlap(self):
        seqs = [[1, 2, 3], [2, 3, 4]]
        merged = _find_longest_common_token_sequence(seqs, set())
        assert merged == [1, 2, 3, 4]

    def test_special_tokens_filtered(self):
        seqs = [[1, 99, 2, 3], [2, 3, 99, 4]]
        special = {99}
        merged = _find_longest_common_token_sequence(seqs, special)
        assert 99 not in merged
        assert merged == [1, 2, 3, 4]

    def test_single_sequence(self):
        seqs = [[10, 20, 30]]
        merged = _find_longest_common_token_sequence(seqs, set())
        assert merged == [10, 20, 30]

    def test_three_sequences(self):
        seqs = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        merged = _find_longest_common_token_sequence(seqs, set())
        assert merged == [1, 2, 3, 4, 5]


class TestStitchStep:
    """The generic LCS stitch shared by the timed and untimed chunked paths."""

    def _config(self):
        return LongformConfig(chunk_duration=30.0, stride=26.0)

    def test_string_items_match_legacy_merge(self):
        from crisperwhisper.longform.chunked_lcs import _stitch_step

        prev = ["a", "b", "brown", "fox", "c"]
        nxt = ["brown", "fox", "jumps", "over"]
        merged, length, lcs_words = _stitch_step(
            prev, nxt, self._config(), word_of=lambda w: w,
        )
        assert merged == ["a", "b", "brown", "fox", "jumps", "over"]
        assert length == 2
        assert lcs_words == "brown fox"

    def test_no_overlap_concatenates(self):
        from crisperwhisper.longform.chunked_lcs import _stitch_step

        merged, length, lcs_words = _stitch_step(
            ["x", "y"], ["p", "q"], self._config(), word_of=lambda w: w,
        )
        assert merged == ["x", "y", "p", "q"]
        assert length == 0
        assert lcs_words == ""

    def test_word_timestamp_items_same_cut_as_strings(self):
        """The stitch decision must be identical whether items are plain
        strings or WordTimestamp objects (it compares word strings)."""
        from crisperwhisper.longform.chunked_lcs import _stitch_step

        prev_s = ["a", "b", "brown", "fox", "c"]
        nxt_s = ["brown", "fox", "jumps"]
        prev_wt = [_wt(i, i + 0.5, w) for i, w in enumerate(prev_s)]
        nxt_wt = [_wt(26 + i, 26 + i + 0.5, w) for i, w in enumerate(nxt_s)]

        merged_s, len_s, _ = _stitch_step(
            prev_s, nxt_s, self._config(), word_of=lambda w: w,
        )
        merged_wt, len_wt, _ = _stitch_step(
            prev_wt, nxt_wt, self._config(), word_of=lambda w: w.word,
        )
        assert len_wt == len_s
        assert [w.word for w in merged_wt] == merged_s
        # kept LCS words carry the PREVIOUS chunk's timings; post-LCS words
        # carry the new chunk's timings
        assert merged_wt[2].start == 2       # "brown" from prev
        assert merged_wt[-1].start == 26 + 2  # "jumps" from nxt


class TestMergeWithProvenance:
    def test_tags_match_merged_tokens(self):
        from crisperwhisper.longform.token_lcs import _merge_with_provenance

        seqs = [[1, 2, 3], [2, 3, 4]]
        merged = _merge_with_provenance(seqs, set())
        assert [t for t, _, _ in merged] == [1, 2, 3, 4]
        # prefix comes from chunk 0, appended suffix from chunk 1
        assert [(c, k) for _, c, k in merged] == [(0, 0), (0, 1), (0, 2), (1, 2)]

    def test_special_filter_keeps_original_indices(self):
        from crisperwhisper.longform.token_lcs import _merge_with_provenance

        # 99 is special: filtered before alignment, but orig indices must
        # still refer to the UNFILTERED sequences.  (The merge needs a >1
        # token overlap to stitch -- same as the plain HF-style merge.)
        seqs = [[99, 1, 2, 3], [2, 99, 3, 4]]
        merged = _merge_with_provenance(seqs, {99})
        assert [t for t, _, _ in merged] == [1, 2, 3, 4]
        assert [(c, k) for _, c, k in merged] == [
            (0, 1), (0, 2), (0, 3), (1, 3),
        ]

    def test_equivalence_with_plain_merge(self):
        from crisperwhisper.longform.token_lcs import (
            _find_longest_common_token_sequence,
            _merge_with_provenance,
        )

        seqs = [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7]]
        plain = _find_longest_common_token_sequence(seqs, {99})
        tagged = _merge_with_provenance(seqs, {99})
        assert plain == [t for t, _, _ in tagged]
