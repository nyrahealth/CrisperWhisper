# CrisperWhisper Documentation

Complete reference for the `crisperwhisper` package. New here? Start with
the [README](README.md) for a quick overview and quickstart, or the
[release post](https://www.nyra-labs.com/crisperwhisper) for what
CrisperWhisper 2.0 is and how it performs.

CrisperWhisper is fast, accurate speech recognition with verbatim
transcription support. It runs on two interchangeable backends: a custom
[CTranslate2](https://github.com/nyrahealth/CTranslate2) fork (fast, with
speculative decoding) or pure HuggingFace Transformers (portable torch).

## Models

CrisperWhisper 2.0 ships in four sizes. Everywhere a model id is
expected, the bare size name works as shorthand
(`CrisperWhisperModel("turbo")`); the default is **large**.

| Model | HuggingFace ID | Notes |
|-------|---------------|-------|
| **large** (default) | `nyralabs/CrisperWhisper2.0_large` | Best open quality |
| **turbo** | `nyralabs/CrisperWhisper2.0_turbo` | Fastest large-quality option (4 decoder layers); also the recommended speculative draft for `large` |
| **medium** | `nyralabs/CrisperWhisper2.0_medium` | |
| **small** | `nyralabs/CrisperWhisper2.0_small` | Smallest; useful as a speculative draft |
| **large_pro** / **turbo_pro** / **medium_pro** / **small_pro** | `nyralabs/CrisperWhisper2.0_<size>_pro` | Pro: Nyra's best models, with improved performance, hotword boosting, trained on additional proprietary data; commercial license only |
| CrisperWhisper (v1) | `nyrahealth/CrisperWhisper` | Verbatim only, HuggingFace Transformers backend (deprecated) |

All sizes support verbatim + intended modes, verbatimize, longform, word
timings, and speculative decoding (ct2 backend). Hotword boosting is a
Pro-model feature.

The standard models are released under a
[non-commercial research license](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md)
and are available for commercial licensing; the Pro models are available
under commercial license only ([contact](https://www.nyra-labs.com/crisperwhisper)).
The inference code in this repository is MIT (see [License](#license)).

## Backends

CrisperWhisper 2 runs on either backend; pick at install time and/or with
the `backend=` argument.

| Capability | `ct2` (CTranslate2) | `transformers` |
|------------|:-------------------:|:--------------:|
| Verbatim / intended modes, hotwords (Pro), verbatimize | yes | yes |
| Word-level timestamps (Viterbi on cross-attention) | yes | yes |
| Longform (continuation + LCS strategies) | yes | yes |
| Hallucination mitigation (rewind/escape repair) | yes | yes |
| Dual-mode (verbatim+intended in one batched pass) | yes | no |
| Speculative decoding | yes | no |
| HF -> CT2 weight conversion | yes | n/a |
| Relative speed | ~4-5x | 1x (baseline) |

Both backends run the *same* timing, longform and hallucination-repair
algorithms.  The `transformers` backend recovers cross-attention for word
timing with eager attention (`output_attentions`), so it loads the model
with `attn_implementation="eager"` (SDPA / flash attention do not expose
attention weights).

## Performance

> Deep dive: [Measuring verbatimness: the Nyra Verbatim Speech Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark)

Benchmarked on NVIDIA L40 GPU, float16 precision, 30-second German
parliamentary speech (Bundestag):

| Backend | RTF | Speedup |
|---------|-----|---------|
| HuggingFace Transformers (fp16) | 0.091 | 1.0x |
| CrisperWhisper 2 (CTranslate2) | 0.023 | **3.9x** |
| CrisperWhisper 2 + Speculative Decoding | 0.018 | **5.3x** |

RTF = processing time / audio duration. Lower is better.

## Installation

The core package installs **no inference backend**. Choose one (or both)
via extras:

```bash
pip install crisperwhisper[ct2]            # CTranslate2 (fast, speculative decoding)
pip install crisperwhisper[transformers]   # pure torch + HuggingFace Transformers
pip install crisperwhisper[all]            # both backends
```

`[ct2]` pulls in `ctranslate2-crisperwhisper`, a drop-in replacement for
CTranslate2 with speculative-decoding APIs.  `[transformers]` pulls in
`torch` + `transformers`.

`[ct2]` GPU support needs only an NVIDIA driver: the CUDA userspace
libraries (cuBLAS 12) arrive via pip and are loaded automatically; no
system CUDA installation or `LD_LIBRARY_PATH` setup. Wheels are Linux
x86_64; on other platforms use `[transformers]`. Do **not** install
`faster-whisper` (or upstream `ctranslate2`) alongside `[ct2]`: upstream
ctranslate2 overwrites the fork's files in site-packages.

For first-time model conversion from HuggingFace format to CT2:

```bash
pip install crisperwhisper[convert]
```

### Docker

```bash
docker run --gpus all nyrahealth/crisperwhisper \
  transcribe audio.wav --language en --mode verbatim
```

## Quick Start

### CrisperWhisper 2 (recommended)

```python
from crisperwhisper import CrisperWhisperModel

# Defaults to nyralabs/CrisperWhisper2.0_large;
# backend="auto" prefers ct2 when installed, else transformers
model = CrisperWhisperModel()

result = model.transcribe("audio.wav", language="en")
print(result.text)

# Or pick a size by shorthand:
model = CrisperWhisperModel("turbo")
```

### Choosing a backend

```python
# Force CTranslate2 (fast, supports speculative decoding)
model = CrisperWhisperModel("large", backend="ct2")

# Force pure-torch Transformers (no speculative decoding)
model = CrisperWhisperModel("large", backend="transformers")

# Word timings, longform and hallucination repair all work the
# same on either backend:
result = model.transcribe("audio.wav", word_timestamps=True)
print(model.backend, [(w.word, w.start, w.end) for w in result.words][:5])
```

### With Speculative Decoding (1.3-1.4x faster, ct2 only)

```python
model = CrisperWhisperModel(
    "large",
    backend="ct2",
    draft_model="turbo",
)

result = model.transcribe("audio.wav", speculative_decoding=True)
```

By default `K` (tokens drafted per round) self-tunes to the audio
(`speculative_k="auto"`); see [Speculative Decoding](#speculative-decoding-ct2-backend-only)
for fixed-K and tuning options. Speculative decoding requires the `ct2`
backend; requesting it on the `transformers` backend emits a warning and
falls back to normal decoding.

### CrisperWhisper v1 (legacy, deprecated)

The original `nyrahealth/CrisperWhisper` is a plain Whisper model with a
changed tokenizer (one explicit space token for sharper word timing).  It
is still supported (on the `transformers` backend, reusing the same
Viterbi timing and hallucination-repair code as v2) but emits a
`DeprecationWarning` on load.  Requires the `[transformers]` extra.

```python
model = CrisperWhisperModel("nyrahealth/CrisperWhisper")  # DeprecationWarning
result = model.transcribe("audio.wav", word_timestamps=True)
for w in result.words:
    print(w.word, w.start, w.end)
```

Differences from v2:

- **Verbatim only**: no intended mode, hotwords, verbatimize, or
  speculative decoding (each ignored with a warning).
- **Word timings** use the explicit space token's cross-attention as the
  pause signal (`blank_source="space"`) instead of mel energy; the model's
  own `generation_config` alignment heads are used by default.
- **No context-aware longform**: audio longer than 30s is transcribed as
  independent 30s windows (a warning is emitted).  Use CrisperWhisper 2 for
  seamless longform.

## Features

### Transcription Modes

> Deep dive: [How we unlocked multilingual style-controlled transcription at scale](https://www.nyra-labs.com/research/multilingual-style-control)

CrisperWhisper 2 transcribes in two styles that differ **only** by the
decoder prompt prefix (the encoder output is identical):

```python
# Verbatim: preserves disfluencies, stutters, filler words ([UH], [UM])
result = model.transcribe("audio.wav", mode="verbatim")

# Intended: clean, fluent transcript
result = model.transcribe("audio.wav", mode="intended")
```

Pick the path that matches what you need:

| You want… | Call | Cost |
|-----------|------|------|
| **Intended only** | `transcribe(..., mode="intended")` | 1 decode |
| **Verbatim only** | `transcribe(..., mode="verbatim")` (default) | 1 decode |
| **Both at once** | `transcribe_dual(...)` | **~1 decode** (both modes batched) |

#### Both modes at once (transcribe_dual, ct2 only)

Because verbatim and intended share the same encoder output and differ
only by the prompt prefix, both can be decoded **together in a single
batched decoder pass**: the encoder runs once and each mode is a row in a
batch-2 greedy decode that runs in lockstep. The expensive autoregressive
decode happens once, so the second transcript is almost free.

```python
verbatim, intended = model.transcribe_dual(
    "audio.wav",
    language="en",
    modes=("verbatim", "intended"),   # default; order = output order
    word_timestamps=True,             # captured inline for each mode
)

print(verbatim.text)
print(intended.text)
for w in verbatim.words:
    print(f"{w.start:6.2f} - {w.end:6.2f}  {w.word}")
```

`transcribe_dual` returns one `TranscriptionResult` per requested mode, in
the order given. It supports everything `transcribe` does: `hotwords`,
`word_timestamps` (cross-attention captured inline during the shared pass,
no extra forward), per-row `hallucination_mitigation`, and longform
(`longform_strategy="continuation"` only). It is roughly **1.9x faster
than two separate `transcribe` calls** (measured ~1.86-1.90x on short and
longform audio; the GPU has ~96% idle decode compute at batch=1, so the
second row costs only ~4% more wall time).

Requirements: the `ct2` backend and a CrisperWhisper 2 model. On the
`transformers` backend or a v1 model it raises `NotImplementedError`; for
longform, only `longform_strategy="continuation"` is accepted.

#### ⚠️ Exactness caveat (fp16 batched-GEMM rounding)

Each row of `transcribe_dual` is *mathematically* identical to the
corresponding standalone `transcribe(mode=...)` call, but **not always
bit-identical** in float16:

- **Short audio (≤30 s): bit-identical.** Both text and word timings match
  per-mode `transcribe()` exactly.
- **Longform (>30 s): rare borderline divergence.** Batching two rows
  through one GEMM rounds differently from a batch-of-one at the ULP
  level. That can flip a *near-tie* token (e.g. drop an "and", swap a
  comma for a period), and because longform feeds each chunk's text back
  as the next chunk's continuation context, such a flip can additionally
  nudge a downstream chunk. In practice this is rare and confined to
  borderline tokens/timings, but if you need byte-for-byte parity with
  `transcribe()` on long audio, run the modes separately.

This is purely a floating-point batching artefact; it is **not** caused by
hallucination repair, which still runs per row (a row that loops falls back
to the exact same single-mode rewind-and-escape repair, so repaired rows
are identical to `transcribe()`). Use `compute_type` higher than `float16`
if you need tighter parity, at a speed cost.

### Hotwords (Pro models only)

Guide the model with domain-specific vocabulary. Hotword boosting is
trained into the **Pro models** only; on the standard models the
`hotwords` argument has no effect on recognition.

```python
model = CrisperWhisperModel("large_pro")
result = model.transcribe("audio.wav", hotwords=["HIPAA", "myocardial", "tachycardia"])
```

### Suppress Tokens

Token suppression works identically on both backends and on every decode
path (greedy, hallucination repair, speculative decoding, temperature
fallback):

- **Default**: the model's `generation_config.suppress_tokens` list is
  applied.  On the `ct2` backend the converter copies
  `generation_config.json` into the converted model directory, so the same
  list drives both backends.
- **Per call**: pass `suppress_tokens=[...]` to `transcribe()` /
  `transcribe_dual()` / `verbatimize()` to override the default for that
  call.  An explicit empty list `[]` disables suppression entirely.
- Negative ids (the HF `-1` "default set" sentinel) are filtered out; pass
  explicit token ids.

```python
# Default: generation_config.suppress_tokens applies.
result = model.transcribe("audio.wav")

# Override per call:
result = model.transcribe("audio.wav", suppress_tokens=[220, 50257])

# Disable suppression entirely:
result = model.transcribe("audio.wav", suppress_tokens=[])
```

Note on the first decoded token: both backends intentionally decode with
begin-of-sequence suppression **disabled** (CT2 `suppress_blank=False`;
the transformers backend clears `begin_suppress_tokens`), so the model may
legitimately emit an empty transcript on silence.  This matches how the
CrisperWhisper 2 models were evaluated.

### Word-Level Timestamps

> Deep dive: [Turning emergent cross-attention into a precise aligner](https://www.nyra-labs.com/research/attention-to-aligner)

Pass `word_timestamps=True` to get a per-word `(start, end)` timeline
alongside the transcript:

```python
result = model.transcribe("audio.wav", word_timestamps=True)

print(result.text)
for w in result.words:
    print(f"{w.start:6.2f} - {w.end:6.2f}  {w.word}")
```

Under the hood this enables CrisperWhisper-style cross-attention
extraction in the CTranslate2 backend, then runs a Viterbi alignment
(with mel-energy-derived blank states) to convert the per-token
attention into word-level start/end seconds.  Works with all
transcription modes (`verbatim` / `intended` / hotwords), with
hallucination repair, and across continuation longform, where chunk-local
timings are mapped to global audio seconds and a seam-monotonization
pass keeps word starts from ever going backwards across chunk
boundaries.

```python
# Verbatim mode + hotwords (Pro model) + word timestamps
result = model.transcribe(
    "interview.wav",
    mode="verbatim",
    hotwords=["HIPAA", "tachycardia"],
    word_timestamps=True,
)

# Longform (>30s) with global timestamps
result = model.transcribe(
    "podcast.wav",
    longform_strategy="continuation",
    word_timestamps=True,
)
```

Notes:

* Cross-attention is captured inside the CTranslate2 thread-pool job
  itself, so the decoder runs as a single C++ loop and the captured
  rows are concatenated + head-averaged on the device and transferred
  in **one** bulk PCIe copy at the end.  Viterbi alignment is
  vectorised over states.  Net result: word timestamps add **no
  measurable wall-time overhead** versus `word_timestamps=False` on
  both short-form (≤30 s) and longform (60 s+) audio.
* By default the alignment heads come from the model's `config.json`
  (copied from the HuggingFace `generation_config`).  Override with
  `alignment_heads=[(layer, head), ...]` if you have a custom selection.
* Only the `continuation` longform strategy supports the seam-smoothing
  pass.
* `word_timestamps=True` **is** now supported together with
  `speculative_decoding=True` (see below).  It is still **not**
  implemented for `longform_strategy` values other than
  `"continuation"`; the LCS-stitched strategies would need a per-chunk
  timing pass with an overlap merge rule, and that call still raises
  `NotImplementedError`.

#### Word timestamps + speculative decoding

When both are enabled, cross-attention is captured from **both** models
(Option B):

* accepted **draft** tokens keep the **draft model's** cross-attention
  (captured for free while drafting, via
  `forward_step_greedy_with_attention`);
* the always-verified token and any verifier **corrections** keep the
  **main model's** cross-attention (captured in the same batched verify
  pass via the new `forward_batch_with_attention` primitive, with no extra
  forward compute);
* **rejected** draft tokens never contribute a row, so the attention
  matrix stays exactly 1-to-1 with the emitted tokens even across
  speculative rollbacks (`truncate_to_step` trims the cached attention
  consistently).

```python
model = CrisperWhisperModel(
    "large",
    draft_model="turbo",
)
result = model.transcribe(
    "audio.wav",
    speculative_decoding=True,
    word_timestamps=True,   # emits a UserWarning (see note)
)
```

Because two models' alignment heads are mixed (and the draft model's
heads are usually less timing-accurate than the main model's), this path
emits a `UserWarning`.  Word **content** and ordering are unaffected
(speculative decoding is output-preserving in strict mode); only the
fine-grained start/end of words that came from accepted draft tokens may
be marginally less precise.  Disable speculative decoding for the most
accurate timings.

**Hallucination repair** works in this mode too: after the speculative
pass, repetition loops are detected (`find_token_loop`) and the tail is
re-decoded on the **main** model with the loop-starter banned (the same
machinery as the non-speculative `generate_with_repair_and_attention`
path).  Rewound tokens' attention rows are dropped and the re-decoded
tail carries main-model rows, so the 1-to-1 token↔row invariant is
preserved.

### Verbatimize

> Deep dive: [Closing the verbatim data gap with Verbatimize](https://www.nyra-labs.com/research/verbatimize)

Transform a clean transcript into verbatim form using audio context:

```python
result = model.verbatimize("audio.wav", transcript="the patient has chest pain")
```

### Forced Alignment

Align a **known transcript** to audio to get word-level timestamps for that
exact text. Works on **every backend** (ct2, transformers, legacy v1) and any
audio length.

```python
result = model.forced_align("audio.wav", "the exact words that were spoken")
for w in result.words:
    print(w.word, w.start, w.end)
```

Provide naturally-cased, punctuated text (the model was trained on regular
transcriptions); for v2, verbatim-style text aligns most closely.

#### How it works (transcribe -> align -> interpolate)

Rather than teacher-forcing the text through the decoder, forced alignment
reuses the transcription pipeline, which makes it robust to pauses, silences
and hold music:

1. The audio is **transcribed** with the normal longform pipeline, producing
   the model's own hypothesis words *with* cross-attention timestamps. The
   timestamps come from the model's own output (where the cross-attention is
   sharp), and pauses are represented for free.
2. The reference transcript is **aligned** to the hypothesis at the word level
   (`difflib.SequenceMatcher`). Matched words inherit the hypothesis timestamp
   directly.
3. Reference words with no hypothesis match (ASR substitutions, fillers the
   model rendered differently, etc.) are **interpolated** across the interval
   between their surrounding matched anchors, proportional to word length.

Because every reference word is bounded by its two neighboring anchors, a word
can never drift far: there is no catastrophic desync, even on long
conversational audio with multi-second pauses. The trade-off is that unmatched
reference words get interpolated (approximate) times rather than direct
acoustic onsets.

```python
result = model.forced_align(
    "long_audio.wav", transcript_text,
    mode="verbatim",                  # transcription mode for the internal pass
    longform_strategy="continuation",
    hallucination_mitigation=True,
)
```

### Longform Transcription

> Deep dive: [Longform transcription with conditional continuation](https://www.nyra-labs.com/research/longform-continuation)

Audio longer than 30 seconds is automatically chunked. Three strategies are
available:

| Strategy | How it works | Trade-offs |
|----------|-------------|------------|
| `"continuation"` (default) | Sequential: each chunk's decoder prompt includes the last K confirmed words from the previous chunk. | Best quality; cannot be parallelised. |
| `"chunked_lcs"` | Independent: all chunks decoded separately, then stitched by longest-common-subsequence at word level in the overlap region. | Parallelisable; slightly lower accuracy at boundaries. |
| `"token_lcs"` | Independent: like `chunked_lcs` but stitching happens at the token level (HuggingFace pipeline style). | Parallelisable; token-level alignment. |

```python
# Continuation context (default, best quality)
result = model.transcribe("long_audio.wav", longform_strategy="continuation")

# Chunked word-level LCS (parallelizable)
result = model.transcribe("long_audio.wav", longform_strategy="chunked_lcs")

# Token-level LCS
result = model.transcribe("long_audio.wav", longform_strategy="token_lcs")
```

#### How continuation works

The model was trained with a context-continuation objective.  Given a
prompt like `{mode_tags} <ctx> last few words <ectx>`, it outputs only
the text that continues beyond those context words.

1. The audio is sliced into overlapping 30-second windows with a
   configurable stride (default 26 s = 4 s overlap).
2. The first chunk is decoded without context.
3. Each subsequent chunk's prompt includes the last `context_words`
   confirmed words from the accumulated transcript.
4. At every non-final chunk boundary, trailing words are dropped to avoid
   partial-word artefacts.  With `timestamp_aware_drop=True` (default) the
   drop is **overlap-aware**: a trailing word is dropped only when its audio
   starts inside the overlap region, i.e. only when the next window actually
   re-covers it (`drop_words` caps how many may be dropped; words the next
   window cannot re-cover are always kept, never lost).  With
   `timestamp_aware_drop=False` the legacy fixed count of `drop_words`
   trailing words is dropped.
5. The final chunk keeps all its words.

Tune longform parameters:

```python
result = model.transcribe(
    "long_audio.wav",
    longform_strategy="continuation",
    stride=26.0,           # seconds between chunks (4s overlap)
    context_words=12,      # words passed as context to next chunk
    drop_words=2,          # cap on words dropped at chunk boundaries
    timestamp_aware_drop=True,  # only drop words the next window re-covers
)
```

These parameters live in `crisperwhisper/longform/base.py` (`LongformConfig`)
and are passed through the `transcribe()` API.

#### Temperature fallback (collapse recovery)

Greedy decoding occasionally *collapses* on a chunk: the model emits a
confident but near-empty transcription (e.g. `"Meanwhile."` for 30 s of
dense speech).  With `temperature_fallback=True` (default, both backends)
each chunk is coverage-checked: when speech clearly fills the audio but
almost no words came out (confirmed against a sibling-mode decode), the
chunk is re-decoded with an escalating temperature ladder
(0.4 → 0.6 → 0.8 → 1.0, several seeded draws each) and the first decode
that covers the audio wins.  See `crisperwhisper/fallback.py`.

Note: the fallback needs the engine's `generate_sampled` primitive, which
the speculative decoder does not expose, so with
`speculative_decoding=True` the fallback is inactive (transcription
proceeds normally without it).

### Speculative Decoding (ct2 backend only)

> Deep dive: [Faster inference and mitigating hallucinations](https://www.nyra-labs.com/research/killing-hallucinations)

Uses a smaller draft model to propose tokens that the main model verifies
in a single batched pass. The custom CTranslate2 fork provides KV-cache
persistence and GPU-side argmax, eliminating the overhead that makes naive
speculative decoding slower.  This feature is **only available on the
`ct2` backend**; on `transformers` it is ignored (with a warning).

```python
model = CrisperWhisperModel(
    "nyrahealth/CrisperWhisper2",
    backend="ct2",
    draft_model="nyrahealth/CrisperWhisper2-turbo",
    # speculative_k="auto" is the default (self-tuning K)
)

# Strict mode (default): output identical to main model alone
result = model.transcribe("audio.wav", speculative_decoding=True)

# Semantic mode: accepts punctuation/casing differences for higher throughput
result = model.transcribe(
    "audio.wav",
    speculative_decoding=True,
    speculative_mode="semantic",
)
```

#### Choosing K (`speculative_k`)

`K` is the number of tokens the draft proposes per verify round. The right
value depends on how often the draft is correct (its acceptance rate),
which varies by audio. You don't have to tune it:

```python
# Self-tuning (default): the model finds a good K on its own:
model = CrisperWhisperModel(..., speculative_k="auto")

# Fixed K: pin it to a constant:
model = CrisperWhisperModel(..., speculative_k=10)
```

- **`"auto"` (default)**: K self-tunes to the draft's acceptance with an
  AIMD controller (additive-increase / additive-decrease, as in HF assisted
  decoding): a round where every drafted token is accepted bumps K up by two;
  any rejection nudges it down by one. The controller's K **persists across
  chunks** of a transcription (and is re-seeded at the start of each new
  audio), so over a file it converges to the acceptance-driven equilibrium,
  roughly the K at which about a third of the rounds fully accept. The +2/−1
  up-bias keeps K near the cap when the draft's acceptance is high (where the
  wall-time optimum sits for `large-v2` + `turbo`) while still backing off on
  low-acceptance audio. K is capped at 16; no window or seed needs hand-tuning.
- **`<int>`**: a fixed K (no adaptation).

Strict speculative decoding is **output-preserving regardless of K**, so
`K` only affects throughput, never the transcript. In benchmarks on
`large-v2` + `turbo`, `"auto"` matches the best hand-picked fixed K on both
high- and low-acceptance audio without any tuning.

Power users can instead supply an explicit adaptive window via
`min_speculative_tokens` / `max_speculative_tokens` (the controller then
self-tunes within those bounds instead of the `"auto"` defaults). The
legacy `num_speculative_tokens=<int>` argument still works as an alias for
a fixed K.

**Speedup by audio length (strict mode):**

| Audio Length | Speedup vs Normal |
|-------------|-------------------|
| < 10s | ~0.8x (overhead dominates) |
| 20-30s | **1.25-1.39x** |
| 30-90s (longform) | **1.24-1.38x** |

Speculative decoding is automatically disabled for short audio where it
would not provide a benefit.

### Quantization

```python
# FP16 (default, fastest on modern GPUs)
model = CrisperWhisperModel("nyrahealth/CrisperWhisper2", compute_type="float16")

# INT8+FP16 (smaller model size, similar speed)
model = CrisperWhisperModel("nyrahealth/CrisperWhisper2", compute_type="int8_float16")
```

### Hallucination Mitigation

> Deep dive: [Faster inference and mitigating hallucinations](https://www.nyra-labs.com/research/killing-hallucinations)

Repetition-loop detection and repair is enabled by default
(`hallucination_mitigation=True`) on **both backends**.  It can be disabled
per call:

```python
result = model.transcribe("audio.wav", hallucination_mitigation=False)
```

The system uses **context repair**: the model decodes freely with no
per-step constraints.  After each greedy pass the output is scanned for
consecutive n-gram repetitions.  When a loop is found the output is
rewound, one "escape" token is forced (the loop-starting token is banned
for that single step), and free decoding resumes.  The `ct2` backend runs
this via `crisperwhisper.hallucination.generate_with_repair`; the
`transformers` backend re-implements the same control flow natively in
`TransformersEngine.generate_with_repair`, reusing the shared loop
detector (`find_token_loop`) and thresholds.

#### Per-ngram thresholds

Different n-gram sizes use different repetition thresholds: short
unigrams require more repeats before triggering a repair than long
phrases which are almost never genuine speech:

```python
# crisperwhisper/hallucination.py
DEFAULT_REPAIR_THRESHOLDS: dict[int, int] = {
    1: 8,   # single tokens: 8 consecutive copies
    2: 8,   # bigrams
    3: 4,   # trigrams
    4: 3,   # 4-grams
    5: 3,   # 5-grams
}
```

To adjust these thresholds globally, edit `DEFAULT_REPAIR_THRESHOLDS` in
`crisperwhisper/hallucination.py`, or pass custom thresholds at call-time
via the lower-level API:

```python
from crisperwhisper.hallucination import generate_with_repair

gen_ids, n_repairs = generate_with_repair(
    engine, features, prompt_tokens,
    detect_reps={1: 10, 2: 6, 3: 3, 4: 2, 5: 2},  # custom thresholds
    keep_reps=1,       # copies kept after rewind
    max_repairs=3,     # max rewind cycles before giving up
)
```

#### Other strategies (advanced)

Two additional strategies exist in `crisperwhisper/hallucination.py`:

- **`generate_with_blocking`**: real-time n-gram blocker that bans the
  loop-starting token at each decoding step.  Used internally when
  step-level control is needed.
- **`find_token_loop`**: post-hoc scanner used as a safety net after
  speculative decoding (where step-by-step blocking is not possible).

## Result Object

```python
result = model.transcribe("audio.wav")

result.text               # full transcript
result.language           # language code
result.mode               # "verbatim", "intended", or "verbatimize"
result.duration           # audio duration in seconds
result.processing_time    # inference time in seconds
result.chunks             # per-chunk details (longform only)
result.words              # list[WordTimestamp] when transcribed with
                          # word_timestamps=True, otherwise None.
                          # v1 models populate this automatically; v2
                          # models populate it only when the flag is set.
```

Each entry of `result.words` is a `WordTimestamp(word: str, start:
float, end: float)`, with `start`/`end` in seconds of the original
audio (already lifted out of chunk-local coordinates for longform).

## Model Conversion

On the `ct2` backend, HuggingFace models are automatically converted to
CTranslate2 format on first load.  Converted models are cached in
`~/.cache/crisperwhisper/` (override with `$CRISPERWHISPER_CACHE`).  The
`transformers` backend loads HuggingFace weights directly and needs no
conversion.

Pass a pre-converted CTranslate2 model directory directly (ct2), or a
HuggingFace id/dir (either backend):

```python
model = CrisperWhisperModel("/path/to/ct2_model", backend="ct2")
model = CrisperWhisperModel("/path/to/hf_model", backend="transformers")
```

## Architecture

```
CrisperWhisperModel          (public API; selects backend)
  ├── backends
  │     ├── CT2Engine         (CTranslate2 runtime; bulk
  │     │                      generate_greedy_with_attention + on-device
  │     │                      concat/head-mean GPU->CPU transfer)
  │     ├── SpeculativeDecoder (draft/main verification with KV-cache; ct2)
  │     └── TransformersEngine (pure torch; cross-attention captured inline
  │                            during generation with eager attention; the
  │                            teacher-forced pass remains only as the
  │                            forced-aligner primitive)
  ├── PromptBuilder          (verbatim/intended/hotword/verbatimize prompts)
  ├── Longform strategies    (continuation, chunked_lcs, token_lcs)
  ├── HallucinationRepair    (repetition detection + regeneration; per
  │                           backend, sharing find_token_loop)
  ├── WordTimingExtractor    (vectorised viterbi alignment of cross-
  │                           attention + mel-energy blank states ->
  │                           WordTimestamp; backend-agnostic)
  └── ModelConverter         (HF → CT2 with custom token handling; ct2)
```

The shared algorithms (prompt building, word timing, longform, repair,
temperature fallback) depend only on a small engine interface,
documented as `crisperwhisper.interfaces.EngineProtocol`, so both
`CT2Engine` and `TransformersEngine` run them unchanged.

## API Reference

### `CrisperWhisperModel`

```python
CrisperWhisperModel(
    model_name_or_path: str,
    *,
    backend: str = "auto",              # "auto" | "ct2" | "transformers"
    compute_type: str = "float16",
    device: str = "auto",
    device_index: int = 0,
    draft_model: str | None = None,     # ct2 only (speculative decoding)
    speculative_k: int | str = "auto",  # ct2 only: "auto" (self-tuning) | <int> (fixed)
    num_speculative_tokens: int | None = None,  # deprecated alias for fixed speculative_k=<int>
    min_speculative_tokens: int = 0,    # ct2 only: explicit adaptive-K window (power users)
    max_speculative_tokens: int = 0,    # ct2 only: explicit adaptive-K window (power users)
    cache_dir: str | Path | None = None,  # ct2 conversion cache
)
```

`speculative_k` controls how many tokens the draft proposes per round.
`"auto"` (default) self-tunes it to the draft's acceptance and persists the
learned value across chunks; pass an int for a fixed K. See
[Speculative Decoding](#speculative-decoding-ct2-backend-only).

### `transcribe()`

```python
model.transcribe(
    audio,                              # file path or numpy array
    *,
    language: str = "en",
    mode: str = "verbatim",             # "verbatim" or "intended"
    hotwords: list[str] | None = None,
    sr: int | None = None,              # sample rate when audio is a numpy array
    longform_strategy: str = "continuation",
    chunk_duration: float = 30.0,       # longform window length (<= 30s)
    stride: float = 26.0,
    context_words: int = 12,
    drop_words: int = 2,
    timestamp_aware_drop: bool = True,  # overlap-aware boundary drop (see Longform)
    temperature_fallback: bool = True,  # coverage-gated collapse recovery
    max_new_tokens: int = 256,
    speculative_decoding: bool = False,
    speculative_mode: str = "strict",   # "strict" or "semantic"
    hallucination_mitigation: bool = True,
    word_timestamps: bool = False,      # populate result.words
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,  # None = generation_config default
) -> TranscriptionResult
```

### `transcribe_dual()`

Decode several modes of one audio in a single batched pass (ct2 + v2 only).
Returns one `TranscriptionResult` per mode, in `modes` order. See
[Both modes at once](#both-modes-at-once-transcribe_dual-ct2-only) for the
~1.9x speedup and the float16 exactness caveat on longform.

```python
model.transcribe_dual(
    audio,                                  # file path or numpy array
    *,
    language: str = "en",
    modes: tuple[str, ...] = ("verbatim", "intended"),
    hotwords: list[str] | None = None,
    sr: int | None = None,
    longform_strategy: str = "continuation",  # only "continuation" supported
    chunk_duration: float = 30.0,
    stride: float = 26.0,
    context_words: int = 12,
    drop_words: int = 2,
    timestamp_aware_drop: bool = True,
    temperature_fallback: bool = True,       # per-row collapse recovery
    max_new_tokens: int = 256,
    hallucination_mitigation: bool = True,   # per-row repair fallback
    word_timestamps: bool = False,           # captured inline per mode
    alignment_heads: list[tuple[int, int]] | None = None,
    suppress_tokens: list[int] | None = None,  # shared across modes
) -> tuple[TranscriptionResult, ...]
```

### `verbatimize()`

```python
model.verbatimize(
    audio,
    transcript: str,
    *,
    language: str = "en",
    sr: int | None = None,
    max_new_tokens: int = 256,
    hallucination_mitigation: bool = True,
    suppress_tokens: list[int] | None = None,
) -> TranscriptionResult
```

### `forced_align()`

```python
model.forced_align(
    audio,
    text: str,
    *,
    language: str = "en",
    mode: str = "verbatim",            # transcription mode for internal pass
    sr: int | None = None,
    longform_strategy: str = "continuation",
    hallucination_mitigation: bool = True,
    alignment_heads: list[tuple[int, int]] | None = None,
) -> TranscriptionResult  # mode="forced_align", result.words populated
```

Transcribes the audio, then aligns the reference `text` to the hypothesis and
interpolates unmatched words. Works on all backends.

## License

The inference code in this repository is **MIT** (see [LICENSE](LICENSE)).

The **model weights are licensed separately**: the standard models under the
[Nyra Health Non-Commercial Research License](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md)
(non-commercial use; commercial licensing available), the Pro models under
commercial license only. See the [README](README.md#license) and the
[model cards](https://huggingface.co/nyralabs).
