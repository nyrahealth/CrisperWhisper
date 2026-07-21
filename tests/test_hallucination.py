"""Tests for hallucination detection (pure-logic, no model needed)."""

import pytest

from crisperwhisper.hallucination import find_token_loop


class TestFindTokenLoop:
    def test_no_loop(self):
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert find_token_loop(ids, reps=3) is None

    def test_unigram_loop(self):
        ids = [10, 20, 5, 5, 5, 5, 5, 5, 5, 5]
        result = find_token_loop(ids, reps=8)
        assert result is not None
        start, gram = result
        assert gram == (5,)
        assert start == 2

    def test_bigram_loop(self):
        ids = [1, 2] * 10
        result = find_token_loop(ids, reps=8)
        assert result is not None
        _, gram = result
        assert len(gram) <= 2

    def test_trigram_loop(self):
        ids = [99] + [3, 4, 5] * 9
        result = find_token_loop(ids, reps=8, max_ngram=5)
        assert result is not None
        _, gram = result
        assert gram == (3, 4, 5)

    def test_threshold_exactly_met(self):
        ids = [7] * 8
        assert find_token_loop(ids, reps=8) is not None

    def test_threshold_not_met(self):
        ids = [7] * 7
        assert find_token_loop(ids, reps=8) is None

    def test_empty_input(self):
        assert find_token_loop([], reps=3) is None

    def test_min_ngram_filter(self):
        # Unigram loop exists but we only search bigrams+
        ids = [5] * 20
        result = find_token_loop(ids, min_ngram=2, reps=8)
        assert result is not None  # 5,5 repeated as bigram

    def test_max_ngram_filter(self):
        ids = [1, 2, 3, 4, 5, 6] * 10
        result = find_token_loop(ids, max_ngram=3, reps=8)
        # 6-gram wouldn't be found with max_ngram=3
        # but sub-patterns might not exist
        # The 6-gram repeats but max_ngram=3 won't catch it
        assert result is None
