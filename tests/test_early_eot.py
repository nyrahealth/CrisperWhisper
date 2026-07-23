"""Tests for context-conditioned early-EOT recovery.

The bulk are pure-Python unit tests of the gate decision logic
(:func:`crisperwhisper.longform.early_eot.recover_early_eot`) driven by a stub
engine -- they run anywhere, no model or GPU.  The GPU / transformers marked
tests exercise the real engine primitives (``eot_probability`` /
``greedy_stops_and_decode``) end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from crisperwhisper.longform.base import EarlyEotConfig
from crisperwhisper.longform.early_eot import (
    _speech_active_after,
    engine_supports_recovery,
    recover_early_eot,
)
from crisperwhisper.result import WordTimestamp


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

EOT = 99


class StubEngine:
    """Engine double for the gate.

    ``eot_probability`` returns the scripted stop probability (and records that
    it was asked -- the gate should only call it after the cheap gap check
    passes).  ``greedy_stops_and_decode`` returns the scripted forced-continuation
    ``(ext_ids, ext_prob)`` and records the ``min_new_tokens`` it was asked for.
    """

    eot_id = EOT

    def __init__(self, ext_ids, ext_prob, *, stop_prob=0.5):
        self._ext_ids = list(ext_ids)
        self._ext_prob = ext_prob
        self._stop_prob = stop_prob
        self.calls: list[int] = []       # greedy_stops_and_decode min_new_tokens
        self.eot_calls: int = 0          # eot_probability invocations

    def eot_probability(self, features, prompt_tokens, gen_ids, *, suppress_tokens=None):
        self.eot_calls += 1
        return self._stop_prob

    def greedy_stops_and_decode(
        self, features, prompt_tokens, *, max_length,
        suppress_tokens=None, min_new_tokens=0,
    ):
        self.calls.append(min_new_tokens)
        return list(self._ext_ids), self._ext_prob


def _mel(total_s: float, speech_end_s: float):
    """Log-mel double: energy 1.0 up to ``speech_end_s``, 0.0 (silence) after.

    So ``_speech_active_after(mel, t)`` ~= ``max(0, speech_end_s - t)``.
    """
    frames = int(total_s * 100)
    hi = int(speech_end_s * 100)
    energy = np.zeros((80, frames), dtype=np.float32)
    energy[:, :hi] = 1.0
    return energy


def _words(*ends):
    return [WordTimestamp(word=f"w{i}", start=None, end=e) for i, e in enumerate(ends)]


def _call(engine, *, gen_ids, word_ts, mel, is_last, config):
    return recover_early_eot(
        engine, features=None, mel=mel, prompt_tokens=[1, 2, 3],
        gen_ids=list(gen_ids), word_ts=word_ts,
        is_last=is_last, max_length=256, suppress_tokens=None, config=config,
    )


# --------------------------------------------------------------------------
# The mel gap helper
# --------------------------------------------------------------------------

def test_speech_active_after_measures_trailing_speech():
    mel = _mel(total_s=30.0, speech_end_s=20.0)
    assert _speech_active_after(mel, 8.0) == pytest.approx(12.0, abs=0.2)
    assert _speech_active_after(mel, 19.0) == pytest.approx(1.0, abs=0.2)
    # nothing after the speech ends
    assert _speech_active_after(mel, 25.0) == pytest.approx(0.0, abs=0.2)


def test_speech_active_after_pure_silence_is_zero():
    silent = np.zeros((80, 3000), dtype=np.float32)
    assert _speech_active_after(silent, 0.0) == 0.0


# --------------------------------------------------------------------------
# Gate decision logic
# --------------------------------------------------------------------------

def test_recovers_on_low_confidence_stop_with_speech_and_confident_continuation():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    gen = [10, 11, 12]  # 3 content tokens
    out = _call(eng, gen_ids=gen, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == list(range(40)) + [EOT]        # extended + trailing EOT
    assert eng.calls == [len(gen) + 1]           # forced one past the stop


def test_reverts_when_continuation_not_confident():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(200)), ext_prob=0.55, stop_prob=0.5)
    gen = [10, 11, 12]
    out = _call(eng, gen_ids=gen, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == gen                            # reverted -- no blowup
    assert eng.calls == [len(gen) + 1]           # it did attempt


def test_reverts_when_continuation_ran_to_max_without_eot():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(200)), ext_prob=None, stop_prob=0.5)
    out = _call(eng, gen_ids=[10, 11, 12], word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]


def test_confident_stop_is_left_alone():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.95)
    out = _call(eng, gen_ids=[10, 11, 12], word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.eot_calls == 1                    # gap passed, so P(EOT) measured
    assert eng.calls == []                       # but never re-decoded


def test_no_trailing_speech_skips_the_eot_pass():
    """The gap-first reorder: no speech left -> P(EOT) is never even measured."""
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    # last word ends at 19.5 s, speech ends 20 s -> gap 0.5 s < tail_min
    out = _call(eng, gen_ids=[10, 11, 12], word_ts=_words(19.5),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.eot_calls == 0                    # cheap gap check dismissed it
    assert eng.calls == []


def test_final_chunk_uses_smaller_tail_floor_than_nonfinal():
    cfg = EarlyEotConfig()  # tail_min_final=2.0, tail_min_nonfinal=4.0
    # gap = 3 s: above the final floor (2), below the non-final floor (4)
    words = _words(17.0)
    mel = _mel(30, 20)
    eng_final = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    out_final = _call(eng_final, gen_ids=[10, 11, 12],
                      word_ts=words, mel=mel, is_last=True, config=cfg)
    assert out_final == list(range(40)) + [EOT]  # fired on the final chunk

    eng_mid = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    out_mid = _call(eng_mid, gen_ids=[10, 11, 12],
                    word_ts=words, mel=mel, is_last=False, config=cfg)
    assert out_mid == [10, 11, 12]               # non-final: 3 s < 4 s floor
    assert eng_mid.eot_calls == 0                # dismissed before the P(EOT) pass
    assert eng_mid.calls == []


def test_disabled_config_is_a_noop():
    cfg = EarlyEotConfig(enabled=False)
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.1)
    out = _call(eng, gen_ids=[10, 11, 12], word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.eot_calls == 0
    assert eng.calls == []


def test_none_stop_prob_is_left_alone():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=None)
    out = _call(eng, gen_ids=[10, 11, 12], word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.eot_calls == 1                    # measured, came back None
    assert eng.calls == []


def test_no_placeable_word_cannot_locate_position():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    out = _call(eng, gen_ids=[10, 11, 12],
                word_ts=_words(None, None), mel=_mel(30, 20),
                is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.eot_calls == 0
    assert eng.calls == []


def test_trailing_eot_in_gen_ids_is_not_double_counted():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99, stop_prob=0.5)
    # gen_ids already carries a trailing EOT -> content length is 3, min_new 4
    _call(eng, gen_ids=[10, 11, 12, EOT], word_ts=_words(8.0),
          mel=_mel(30, 20), is_last=True, config=cfg)
    assert eng.calls == [4]


# --------------------------------------------------------------------------
# Config validation + feature detection
# --------------------------------------------------------------------------

def test_config_rejects_confident_below_threshold():
    with pytest.raises(ValueError):
        EarlyEotConfig(stop_prob_threshold=0.8, confident_prob=0.6)


def test_config_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        EarlyEotConfig(stop_prob_threshold=0.0)
    with pytest.raises(ValueError):
        EarlyEotConfig(confident_prob=1.5)


def test_engine_supports_recovery_feature_detection():
    # StubEngine provides both primitives.
    assert engine_supports_recovery(StubEngine([], None)) is True

    class OnlyProb:
        def eot_probability(self, *a, **k):
            return 0.5

    assert engine_supports_recovery(OnlyProb()) is False  # missing the re-decode

    class OnlyDecode:
        def greedy_stops_and_decode(self, *a, **k):
            return [], None

    assert engine_supports_recovery(OnlyDecode()) is False  # missing P(EOT)

    class Bare:
        pass

    assert engine_supports_recovery(Bare()) is False


# --------------------------------------------------------------------------
# End-to-end: real engine primitives (skipped without a model)
# --------------------------------------------------------------------------

@pytest.mark.gpu
def test_ct2_primitives_force_past_stop_and_report_confidence(model, en_audio):
    _assert_primitives(model, en_audio)


@pytest.mark.transformers
def test_transformers_primitives_force_past_stop_and_report_confidence(
    tf_model, en_audio
):
    _assert_primitives(tf_model, en_audio)


def _assert_primitives(m, en_audio):
    """eot_probability in [0,1]; min_new_tokens forces a strictly longer decode."""
    from crisperwhisper.audio import load_audio
    from crisperwhisper.prompt import PromptBuilder

    engine = m._engine
    assert engine_supports_recovery(engine)
    audio = load_audio(en_audio)[: 30 * 16000]
    features = engine.extract_features(audio)
    prompt = PromptBuilder(engine, language="en").verbatim(context=None)

    base_ids, base_p = engine.greedy_stops_and_decode(features, prompt, max_length=256)
    assert base_ids, "expected a non-empty decode"
    # a natural stop reports a probability in [0, 1]
    if base_p is not None:
        assert 0.0 <= base_p <= 1.0

    # forcing past the stop yields a strictly longer decode
    forced_ids, _ = engine.greedy_stops_and_decode(
        features, prompt, max_length=256, min_new_tokens=len(base_ids) + 5,
    )
    assert len(forced_ids) > len(base_ids)

    # eot_probability agrees the base decode stopped fairly confidently
    p = engine.eot_probability(features, prompt, base_ids)
    assert p is None or 0.0 <= p <= 1.0
