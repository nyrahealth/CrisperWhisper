"""Word / token confidences (``word_confidence=True``) without a model."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from crisperwhisper.result import TokenProbability, WordTimestamp
from crisperwhisper.word_timing import (
    extract_word_timings,
    monotonize_words,
    word_token_probabilities,
)

EOT = 50257
PIECES = {1: " Hello", 2: " wor", 3: "ld", 4: " again", 220: " ", EOT: "<|endoftext|>"}


class _StubEngine:
    eot_id = EOT

    def __init__(self):
        self.tokenizer = SimpleNamespace(
            decode=lambda ids: "".join(PIECES[int(t)] for t in ids),
        )

    def decode_tokens(self, token_ids, skip_special=True):
        return "".join(
            PIECES[int(t)] for t in token_ids
            if not (skip_special and int(t) == EOT)
        )


def test_word_probability_is_mean_token_probability():
    ids = [1, 2, 3, EOT]
    pieces = [PIECES[t] for t in ids]
    lps = [math.log(0.9), math.log(0.5), math.log(0.8), math.log(0.99)]
    out = word_token_probabilities(ids, pieces, lps, [[0], [1, 2], []])

    (p_hello, toks_hello), (p_world, toks_world), (p_empty, toks_empty) = out
    assert p_hello == pytest.approx(0.9)
    assert p_world == pytest.approx((0.5 + 0.8) / 2)
    assert toks_world == [
        TokenProbability(token=" wor", token_id=2, probability=pytest.approx(0.5)),
        TokenProbability(token="ld", token_id=3, probability=pytest.approx(0.8)),
    ]
    assert p_empty is None and toks_empty == []


def test_word_token_probabilities_rejects_misaligned_input():
    with pytest.raises(ValueError):
        word_token_probabilities([1, 2], [" a", " b"], [0.0], [[0], [1]])


def _attention_for(n_tokens: int, f_enc: int = 1500) -> np.ndarray:
    attn = np.full((n_tokens, f_enc), 1e-4, dtype=np.float32)
    for k in range(n_tokens):
        attn[k, 40 + 30 * k: 50 + 30 * k] = 1.0
    return attn / attn.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("keep_unplaceable", [False, True])
def test_extract_word_timings_attaches_confidences(keep_unplaceable):
    ids = [1, 2, 3, 220, 4, EOT]
    lps = np.log(np.array([0.9, 0.5, 0.8, 0.7, 0.6, 0.95], dtype=np.float32))
    mel = np.random.default_rng(0).normal(size=(128, 3000)).astype(np.float32)

    words = extract_word_timings(
        _StubEngine(), ids, _attention_for(len(ids)), mel,
        token_logprobs=lps, keep_unplaceable=keep_unplaceable,
    )
    by_word = {w.word: w for w in words}
    assert set(by_word) <= {"Hello", "world", "again"}
    if keep_unplaceable:
        assert [w.word for w in words] == ["Hello", "world", "again"]
    expected = {"Hello": 0.9, "world": (0.5 + 0.8) / 2, "again": 0.6}
    for word, w in by_word.items():
        assert w.probability == pytest.approx(expected[word], rel=1e-5)
        assert sum(t.probability for t in w.tokens) / len(w.tokens) == pytest.approx(
            w.probability, rel=1e-5,
        )
    if "world" in by_word:
        assert [t.token_id for t in by_word["world"].tokens] == [2, 3]


def test_special_tokens_folded_into_a_word_are_not_scored():
    class _Ct2LikeEngine(_StubEngine):
        all_special_ids = {EOT}

        def __init__(self):
            super().__init__()
            pieces = {**PIECES, EOT: ""}
            self.tokenizer = SimpleNamespace(
                decode=lambda ids: "".join(pieces[int(t)] for t in ids),
            )

    ids = [1, 4, EOT]
    lps = np.log(np.array([0.9, 0.6, 0.5], dtype=np.float32))
    mel = np.zeros((128, 3000), dtype=np.float32)
    words = extract_word_timings(
        _Ct2LikeEngine(), ids, _attention_for(len(ids)), mel,
        token_logprobs=lps, keep_unplaceable=True,
    )
    assert words[-1].word == "again"
    assert words[-1].probability == pytest.approx(0.6)
    assert [t.token_id for t in words[-1].tokens] == [4]


def test_extract_word_timings_without_logprobs_leaves_fields_unset():
    ids = [1, 2, 3, EOT]
    mel = np.zeros((128, 3000), dtype=np.float32)
    words = extract_word_timings(
        _StubEngine(), ids, _attention_for(len(ids)), mel, keep_unplaceable=True,
    )
    assert words and all(w.probability is None and w.tokens is None for w in words)


def test_monotonize_keeps_confidences():
    toks = [TokenProbability(" b", 7, 0.5)]
    words = [
        WordTimestamp("a", 0.0, 1.0, probability=0.9),
        WordTimestamp("b", 0.5, 1.5, probability=0.5, tokens=toks),
    ]
    monotonize_words(words)
    assert words[1].start == 1.0
    assert words[1].probability == 0.5 and words[1].tokens == toks


def test_blocked_step_logprob_uses_raw_logits(monkeypatch):
    from crisperwhisper import hallucination

    raw = np.array([[2.0, 1.0, 0.5, -1.0]], dtype=np.float32)
    monkeypatch.setattr(hallucination, "_to_fp32_cpu", lambda _l: raw.copy())
    logits = SimpleNamespace(shape=raw.shape)

    tok, lp = hallucination._argmax_with_bans_and_logprob(
        logits, np.array([1], dtype=np.intp), {0},
    )
    assert tok == 2
    log_z = np.log(np.exp(raw[0].astype(np.float64)).sum())
    assert lp == pytest.approx(0.5 - log_z)
    assert tok == hallucination._argmax_with_bans(
        logits, np.array([1], dtype=np.intp), {0},
    )


# ---------------------------------------------------------------------------
# Repair carries log-probabilities 1-to-1 with the ids (CT2 orchestration).
# ---------------------------------------------------------------------------

PROMPT = [100, 101, 102]
LOOP_TOKEN = 9


def _logprob(abs_pos: int, token: int) -> float:
    """What a teacher-forced pass would give ``token`` at ``abs_pos``."""
    return -(abs_pos * 0.01 + token * 1e-4)


class _FakeState:
    def __init__(self, start_pos, ids, with_lps):
        self.rows = [np.full(4, start_pos + k, dtype=np.float32) for k in range(len(ids))]
        self.collected_logprobs = (
            [_logprob(start_pos + k, t) for k, t in enumerate(ids)] if with_lps else []
        )


class _FakeCT2Engine:
    """Greedy decode that loops on ``LOOP_TOKEN`` until it is banned once."""

    eot_id = EOT

    def __init__(self):
        self.model = SimpleNamespace(encode=lambda f: f)

    def _resolve_suppress(self, suppress):
        return list(suppress or [])

    def generate_greedy_with_attention(
        self, encoded, prefix, *, max_new_tokens, suppress_tokens,
        ban_first_tokens, return_token_logprobs=False,
    ):
        if ban_first_tokens:
            ids = [4, 5, EOT]
        else:
            ids = [1, 2, 3] + [LOOP_TOKEN] * 20 + [EOT]
        ids = ids[:max_new_tokens]
        return _FakeState(len(prefix), ids, return_token_logprobs), ids

    def decode_attention(self, state):
        return np.stack(state.rows) if state.rows else np.zeros((0, 0), np.float32)


@pytest.mark.parametrize("return_logprobs", [False, True])
def test_repair_keeps_logprobs_aligned_with_ids(return_logprobs):
    from crisperwhisper.hallucination import generate_with_repair_and_attention

    out = generate_with_repair_and_attention(
        _FakeCT2Engine(), "features", PROMPT, max_length=64,
        return_logprobs=return_logprobs,
    )
    gen_ids, attn, n_repairs = out[:3]
    assert n_repairs == 1
    assert gen_ids == [1, 2, 3, LOOP_TOKEN, 4, 5, EOT]
    # Attention row ``k`` encodes the absolute position it was produced at.
    assert [int(r[0]) for r in attn] == [len(PROMPT) + k for k in range(len(gen_ids))]
    if not return_logprobs:
        assert len(out) == 3
        return
    lps = out[3]
    expected = [_logprob(len(PROMPT) + k, t) for k, t in enumerate(gen_ids)]
    np.testing.assert_allclose(lps, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end (real model): CT2 needs CW2_MODEL_PATH, transformers
# CW2_HF_MODEL_PATH.
# ---------------------------------------------------------------------------

def _check_confidences(result, reference):
    assert result.text == reference.text
    assert [(w.word, w.start, w.end) for w in result.words] == [
        (w.word, w.start, w.end) for w in reference.words
    ]
    assert all(w.probability is None for w in reference.words)
    for w in result.words:
        assert 0.0 < w.probability <= 1.0
        assert w.tokens
        assert sum(t.probability for t in w.tokens) / len(w.tokens) == pytest.approx(
            w.probability, rel=1e-4,
        )


@pytest.mark.gpu
def test_ct2_inline_logprobs_match_teacher_forced(model, en_audio):
    from crisperwhisper.audio import load_audio
    from crisperwhisper.prompt import PromptBuilder

    eng = model._engine
    features, _mel = eng.extract_features_with_mel(load_audio(en_audio))
    prompt = PromptBuilder(eng, language="en").verbatim()
    eng.enable_attention(None)
    encoded = eng.model.encode(features)
    plain_state, plain_ids = eng.generate_greedy_with_attention(
        encoded, prompt, max_new_tokens=256,
    )
    for ban in ([], [plain_ids[0]]):
        state, ids = eng.generate_greedy_with_attention(
            encoded, prompt, max_new_tokens=256, ban_first_tokens=ban,
            return_token_logprobs=True,
        )
        inline = np.asarray(state.collected_logprobs, dtype=np.float32)
        assert len(inline) == len(ids) == len(state.collected_attention)
        # Raw-logit semantics: the banned first step still matches a
        # teacher-forced pass, which applies no ban.
        teacher = eng.token_logprobs_for_tokens(features, prompt, ids)
        np.testing.assert_allclose(inline, teacher, atol=0.05)
    assert len(plain_state.collected_logprobs) == 0


@pytest.mark.gpu
def test_ct2_transcribe_word_confidence(model, en_audio):
    ref = model.transcribe(en_audio, word_timestamps=True)
    _check_confidences(model.transcribe(en_audio, word_confidence=True), ref)


@pytest.mark.gpu
def test_ct2_longform_word_confidence(model, samples_dir):
    from crisperwhisper.audio import load_audio

    audio = np.concatenate([
        load_audio(str(samples_dir / "example_en_1.wav")),
        load_audio(str(samples_dir / "example_en_2.wav")),
    ])
    ref = model.transcribe(audio, sr=16000, word_timestamps=True)
    res = model.transcribe(audio, sr=16000, word_confidence=True)
    assert res.chunks and len(res.chunks) > 1
    _check_confidences(res, ref)


@pytest.mark.gpu
def test_ct2_dual_inline_logprobs_match_teacher_forced(model, en_audio):
    from crisperwhisper.audio import load_audio
    from crisperwhisper.prompt import PromptBuilder

    eng = model._engine
    features = eng.extract_features(load_audio(en_audio))
    builder = PromptBuilder(eng, language="en")
    # Different prompt lengths exercise the per-row catch-up before lockstep.
    prompts = [builder.verbatim(hotwords=["CrisperWhisper"]), builder.intended()]
    assert len(prompts[0]) != len(prompts[1])
    eng.enable_attention(None)
    kwargs = dict(
        max_new_tokens=256, eot_id=eng.eot_id,
        suppress_tokens=eng._resolve_suppress(None), want_attention=True,
    )

    plain = eng.model.generate_dual_greedy(features, prompts, **kwargs)
    out = eng.model.generate_dual_greedy(
        features, prompts, return_token_logprobs=True, **kwargs,
    )
    assert len(plain) == 2 and len(out) == 3
    for row, (ids, ref_ids) in enumerate(zip(out[0], plain[0])):
        assert list(ids) == list(ref_ids)
        np.testing.assert_array_equal(np.array(out[1][row]), np.array(plain[1][row]))
        assert len(out[2][row]) == len(ids)
        teacher = eng.token_logprobs_for_tokens(features, prompts[row], list(ids))
        np.testing.assert_allclose(out[2][row], teacher, atol=0.05)

    gens, attns, lps = eng.generate_dual_greedy(
        features, prompts, max_length=256, word_timestamps=True,
        return_logprobs=True,
    )
    ref_gens, _ = eng.generate_dual_greedy(
        features, prompts, max_length=256, word_timestamps=True,
    )
    assert gens == ref_gens
    for g, a, lp in zip(gens, attns, lps):
        assert len(lp) == len(g) == len(a)
        assert np.all(lp <= 0.0)


def _assert_same_confidences(dual_words, single_words):
    assert [w.word for w in dual_words] == [w.word for w in single_words]
    np.testing.assert_allclose(
        [w.probability for w in dual_words],
        [w.probability for w in single_words],
        atol=0.05,
    )


@pytest.mark.gpu
def test_ct2_transcribe_dual_word_confidence(model, en_audio):
    refs = model.transcribe_dual(en_audio, word_timestamps=True)
    results = model.transcribe_dual(en_audio, word_confidence=True)
    for res, ref, mode in zip(results, refs, ("verbatim", "intended")):
        _check_confidences(res, ref)
        single = model.transcribe(en_audio, mode=mode, word_confidence=True)
        _assert_same_confidences(res.words, single.words)


@pytest.mark.gpu
def test_ct2_longform_dual_word_confidence(model, samples_dir):
    from crisperwhisper.audio import load_audio

    audio = np.concatenate([
        load_audio(str(samples_dir / "example_en_1.wav")),
        load_audio(str(samples_dir / "example_en_2.wav")),
    ])
    refs = model.transcribe_dual(audio, sr=16000, word_timestamps=True)
    results = model.transcribe_dual(audio, sr=16000, word_confidence=True)
    for res, ref, mode in zip(results, refs, ("verbatim", "intended")):
        assert res.chunks and len(res.chunks) > 1
        _check_confidences(res, ref)
        single = model.transcribe(
            audio, sr=16000, mode=mode, word_confidence=True,
        )
        _assert_same_confidences(res.words, single.words)


@pytest.mark.gpu
def test_ct2_verbatimize_word_confidence(model, en_audio):
    intended = model.transcribe(en_audio, mode="intended").text
    ref = model.verbatimize(en_audio, intended, word_timestamps=True)
    res = model.verbatimize(en_audio, intended, word_confidence=True)
    assert res.words
    _check_confidences(res, ref)


@pytest.mark.transformers
def test_transformers_matches_ct2_confidences(tf_model, model, en_audio):
    res_tf = tf_model.transcribe(en_audio, word_confidence=True)
    _check_confidences(
        res_tf, tf_model.transcribe(en_audio, word_timestamps=True),
    )
    res_ct2 = model.transcribe(en_audio, word_confidence=True)
    if [w.word for w in res_tf.words] == [w.word for w in res_ct2.words]:
        np.testing.assert_allclose(
            [w.probability for w in res_tf.words],
            [w.probability for w in res_ct2.words],
            atol=0.05,
        )
