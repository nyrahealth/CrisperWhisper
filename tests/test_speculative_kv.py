"""Speculative decoding with KV-cache persistence (GPU end-to-end).

Covers:
  * low-level incremental API (prefill / forward_step / forward_batch),
  * speculative-decoding exactness (float32) and parameter forwarding,
  * the user-facing fp16 guarantee (repaired speculative == non-speculative),
  * a perf smoke benchmark.

All tests are skipped unless ``CW2_MODEL_PATH`` points at a converted CT2
model (see ``conftest.py``).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _argmax_1(sv) -> int:
    """Single-position argmax matching speculative.py's ``_argmax``."""
    import ctranslate2

    arr = np.array(
        sv.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
    )
    return int(arr.reshape(-1, arr.shape[-1])[0].argmax())


def _batch_argmax(sv) -> list[int]:
    """Batch argmax matching speculative.py's ``_batch_argmax``."""
    import ctranslate2

    arr = np.array(
        sv.to(ctranslate2.DataType.float32).to_device(ctranslate2.Device.cpu)
    )
    if arr.ndim == 3:
        arr = arr[0]
    return arr.argmax(axis=-1).tolist()


def _strip_trailing_eot(ids: list[int], eot_id: int) -> list[int]:
    """Drop a single trailing EOT token.

    ``engine.generate()`` and ``SpeculativeDecoder.generate()`` differ only
    in whether the terminating EOT is kept in the returned id list; the
    decoded transcript is identical.  Normalising it lets us assert exact
    token-for-token equality of the actual content.
    """
    return ids[:-1] if ids and ids[-1] == eot_id else ids


@pytest.fixture(scope="module")
def engine_fp16(main_model_path):
    from crisperwhisper.engine import CT2Engine

    return CT2Engine(main_model_path, device="cuda", compute_type="float16")


@pytest.fixture(scope="module")
def engine_fp32(main_model_path):
    """A float32 engine (re-converts the model to float32 if needed).

    Used to assert speculative *exactness* without fp16 tie ambiguity.
    """
    from crisperwhisper.engine import CT2Engine
    from crisperwhisper.converter import ensure_ct2_model

    ct2_path = ensure_ct2_model(main_model_path, quantization="float32")
    return CT2Engine(str(ct2_path), device="cuda", compute_type="float32")


def test_low_level_api(engine_fp16, en_audio):
    """prefill + forward_step must reproduce ``generate``."""
    import soundfile as sf
    from crisperwhisper.prompt import PromptBuilder

    engine = engine_fp16
    prompt = PromptBuilder(engine, language="en").verbatim()
    audio, _ = sf.read(en_audio, dtype="float32")
    features = engine.extract_features(audio)

    ref_ids = engine.generate(features, [prompt], max_length=50)[0]

    state, logits = engine.model.prefill(features, prompt)
    next_token = _argmax_1(logits)
    generated = [next_token]
    for _ in range(49):
        step_logits = engine.model.forward_step(state, next_token)
        next_token = _argmax_1(step_logits)
        generated.append(next_token)
        if next_token == engine.eot_id:
            break

    assert ref_ids[: len(generated)] == generated[: len(ref_ids)], (
        "step-by-step generation diverged from generate()"
    )


def test_forward_batch(engine_fp16, en_audio):
    """forward_batch greedy predictions must match sequential forward_step."""
    import soundfile as sf
    from crisperwhisper.prompt import PromptBuilder

    engine = engine_fp16
    prompt = PromptBuilder(engine, language="en").verbatim()
    audio, _ = sf.read(en_audio, dtype="float32")
    features = engine.extract_features(audio)

    state1, logits0 = engine.model.prefill(features, prompt)
    tokens = [_argmax_1(logits0)]
    step_argmaxes = []
    for _ in range(4):
        sl = engine.model.forward_step(state1, tokens[-1])
        step_argmaxes.append(_argmax_1(sl))
        tokens.append(step_argmaxes[-1])

    state2, _ = engine.model.prefill(features, prompt)
    batch_preds = _batch_argmax(engine.model.forward_batch(state2, tokens[:5]))

    assert batch_preds[:4] == step_argmaxes, (
        f"forward_batch {batch_preds[:4]} != forward_step {step_argmaxes}"
    )


def test_speculative_correctness(engine_fp32, en_audio):
    """Same-model strict speculative decoding is mathematically exact:
    token-for-token identical to plain greedy.

    Asserted in **float32**.  In float16 the batched verify pass
    (``forward_batch``) and sequential ``forward_step`` can pick different
    argmaxes at logit *ties* -- which only happens inside degenerate
    repetition loops (e.g. a verbatim ``"m- m- m-"`` runaway).  Those loops
    never survive the user-facing path, which always runs hallucination
    repair (see ``test_speculative_matches_greedy_fp16``).
    """
    import soundfile as sf
    from crisperwhisper.speculative import SpeculativeDecoder
    from crisperwhisper.prompt import PromptBuilder

    engine = engine_fp32
    spec = SpeculativeDecoder(engine, engine, num_speculative_tokens=5)
    pb = PromptBuilder(engine, language="en")
    audio, _ = sf.read(en_audio, dtype="float32")
    features = engine.extract_features(audio)

    for prompt in (pb.verbatim(), pb.intended()):
        normal_ids = engine.generate(features, [prompt], max_length=256)[0]
        spec_ids = spec.generate(features, [prompt], max_length=256)[0]
        eot = engine.eot_id
        assert _strip_trailing_eot(normal_ids, eot) == _strip_trailing_eot(spec_ids, eot), (
            "speculative output diverged from greedy in float32:\n"
            f"  normal: {engine.decode_tokens(normal_ids, skip_special=True)!r}\n"
            f"  spec:   {engine.decode_tokens(spec_ids, skip_special=True)!r}"
        )


def test_param_forwarding(engine_fp32, en_audio):
    """``num_speculative_tokens`` (K) changes only *how many* tokens are
    drafted per round, never the result: output is identical for any K with
    a same-model draft.  Asserted in float32 (see
    ``test_speculative_correctness`` for the fp16 tie caveat).
    """
    import soundfile as sf
    from crisperwhisper.speculative import SpeculativeDecoder
    from crisperwhisper.prompt import PromptBuilder

    engine = engine_fp32
    spec = SpeculativeDecoder(engine, engine, num_speculative_tokens=3)
    prompt = PromptBuilder(engine, language="en").verbatim()
    audio, _ = sf.read(en_audio, dtype="float32")
    features = engine.extract_features(audio)

    ids_k3 = spec.generate(features, [prompt], max_length=256)[0]
    ids_k8 = spec.generate(
        features, [prompt], max_length=256, num_speculative_tokens=8,
    )[0]

    eot = engine.eot_id
    assert _strip_trailing_eot(ids_k3, eot) == _strip_trailing_eot(ids_k8, eot), (
        "different K produced different output (same-model draft)"
    )


def test_speculative_matches_greedy_fp16(main_model_path, en_audio):
    """User-facing guarantee in float16: with hallucination repair on (the
    default), speculative decoding yields the *same transcript* as
    non-speculative decoding.

    Repair removes the degenerate repetition loops that are the only place
    fp16 logit ties can flip the batched verify pass, so the end-to-end
    texts match exactly.
    """
    from crisperwhisper import CrisperWhisperModel

    # Same model as its own draft keeps strict speculative output-exact.
    model = CrisperWhisperModel(main_model_path, draft_model=main_model_path)
    plain = model.transcribe(en_audio, language="en")
    spec = model.transcribe(en_audio, language="en", speculative_decoding=True)

    assert plain.text.strip() == spec.text.strip(), (
        "speculative vs non-speculative transcript mismatch (repair on):\n"
        f"  plain: {plain.text!r}\n  spec:  {spec.text!r}"
    )


def test_performance(engine_fp16, en_audio):
    """Perf smoke test: same-model speculative runs and returns output.

    With a same-model draft there is no real speedup (every drafted token
    is verified by an identical model); this just exercises the timing path
    and asserts non-empty output.
    """
    import soundfile as sf
    from crisperwhisper.speculative import SpeculativeDecoder
    from crisperwhisper.prompt import PromptBuilder

    engine = engine_fp16
    spec = SpeculativeDecoder(engine, engine, num_speculative_tokens=5)
    prompt = PromptBuilder(engine, language="en").verbatim()
    audio, _ = sf.read(en_audio, dtype="float32")
    features = engine.extract_features(audio)

    engine.generate(features, [prompt], max_length=256)  # warmup

    t0 = time.perf_counter()
    normal_ids = engine.generate(features, [prompt], max_length=256)[0]
    t_normal = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec_ids = spec.generate(features, [prompt], max_length=256)[0]
    t_spec = time.perf_counter() - t0

    print(
        f"  normal={t_normal:.3f}s ({len(normal_ids)} tok)  "
        f"spec={t_spec:.3f}s ({len(spec_ids)} tok)"
    )
    assert len(normal_ids) > 0 and len(spec_ids) > 0
