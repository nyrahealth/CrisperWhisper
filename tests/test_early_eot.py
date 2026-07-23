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
    """Engine double: scripts the forced-continuation decode the gate performs.

    ``greedy_stops_and_decode`` ignores the audio and returns the configured
    ``(ext_ids, ext_prob)``, recording the ``min_new_tokens`` it was asked for.
    """

    eot_id = EOT

    def __init__(self, ext_ids, ext_prob):
        self._ext_ids = list(ext_ids)
        self._ext_prob = ext_prob
        self.calls: list[int] = []

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


def _call(engine, *, gen_ids, stop_prob, word_ts, mel, is_last, config):
    return recover_early_eot(
        engine, features=None, mel=mel, prompt_tokens=[1, 2, 3],
        gen_ids=list(gen_ids), stop_prob=stop_prob, word_ts=word_ts,
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
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    gen = [10, 11, 12]  # 3 content tokens
    out = _call(eng, gen_ids=gen, stop_prob=0.5, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == list(range(40)) + [EOT]        # extended + trailing EOT
    assert eng.calls == [len(gen) + 1]           # forced one past the stop


def test_reverts_when_continuation_not_confident():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(200)), ext_prob=0.55)  # long but unsure
    gen = [10, 11, 12]
    out = _call(eng, gen_ids=gen, stop_prob=0.5, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == gen                            # reverted -- no blowup
    assert eng.calls == [len(gen) + 1]           # it did attempt


def test_reverts_when_continuation_ran_to_max_without_eot():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(200)), ext_prob=None)  # never stopped
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=0.5, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]


def test_confident_stop_is_left_alone():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=0.95, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.calls == []                       # never re-decoded


def test_no_trailing_speech_does_not_fire():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    # last word ends at 19.5 s, speech ends 20 s -> gap 0.5 s < tail_min
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=0.5, word_ts=_words(19.5),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.calls == []


def test_final_chunk_uses_smaller_tail_floor_than_nonfinal():
    cfg = EarlyEotConfig()  # tail_min_final=2.0, tail_min_nonfinal=4.0
    # gap = 3 s: above the final floor (2), below the non-final floor (4)
    words = _words(17.0)
    mel = _mel(30, 20)
    eng_final = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out_final = _call(eng_final, gen_ids=[10, 11, 12], stop_prob=0.5,
                      word_ts=words, mel=mel, is_last=True, config=cfg)
    assert out_final == list(range(40)) + [EOT]  # fired on the final chunk

    eng_mid = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out_mid = _call(eng_mid, gen_ids=[10, 11, 12], stop_prob=0.5,
                    word_ts=words, mel=mel, is_last=False, config=cfg)
    assert out_mid == [10, 11, 12]               # non-final: 3 s < 4 s floor
    assert eng_mid.calls == []


def test_disabled_config_is_a_noop():
    cfg = EarlyEotConfig(enabled=False)
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=0.1, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.calls == []


def test_none_stop_prob_is_left_alone():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=None, word_ts=_words(8.0),
                mel=_mel(30, 20), is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.calls == []


def test_no_placeable_word_cannot_locate_position():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    out = _call(eng, gen_ids=[10, 11, 12], stop_prob=0.5,
                word_ts=_words(None, None), mel=_mel(30, 20),
                is_last=True, config=cfg)
    assert out == [10, 11, 12]
    assert eng.calls == []


def test_trailing_eot_in_gen_ids_is_not_double_counted():
    cfg = EarlyEotConfig()
    eng = StubEngine(ext_ids=list(range(40)), ext_prob=0.99)
    # gen_ids already carries a trailing EOT -> content length is 3, min_new 4
    _call(eng, gen_ids=[10, 11, 12, EOT], stop_prob=0.5, word_ts=_words(8.0),
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
    assert engine_supports_recovery(StubEngine([], None)) is False  # no eot_probability

    class Full(StubEngine):
        def eot_probability(self, *a, **k):
            return 0.5

    assert engine_supports_recovery(Full([], None)) is True

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
