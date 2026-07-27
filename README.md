# CrisperWhisper 2.0

[![PyPI](https://img.shields.io/pypi/v/crisperwhisper?cacheSeconds=3600)](https://pypi.org/project/crisperwhisper/)

**The most accurate verbatim speech recognition you can run in production:
controllable, multilingual, and timed to the word.**

[Release post](https://www.nyra-labs.com/crisperwhisper) ·
[Paper](https://arxiv.org/abs/2607.18934) ·
[Full documentation](DOCS.md) ·
[Models](https://huggingface.co/nyralabs) ·
[Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark)

<a href="https://www.nyra-labs.com/crisperwhisper">
  <img src="https://img.shields.io/badge/%E2%96%B6%20%20TRY%20IT%20NOW-2563eb?style=for-the-badge" alt="Try it now">
</a>

Most speech-to-text systems never actually decide whether to write down what
was *said* or what was *meant*. They inherit that choice from their training
data and apply it inconsistently. CrisperWhisper 2.0 makes it an explicit,
controllable choice. One recording, two transcripts:

> **Verbatim**, exactly what was said, in one consistent format:
> `[um] so we we need to, to reschedule the th- thursday meeting to [uh] march third at nine thirty [laughter]`
>
> **Intended**, the clean version the speaker meant, with numbers, dates,
> and emails formatted the way you'd write them:
> `So we need to reschedule the Thursday meeting to March 3 at 9:30.`

On top of that:

- **Word-level timings.** Around 30 ms mean boundary error on read speech
  and 41 ms on conversational speech, the most precise word timing of any
  system we benchmarked, on both.
- **Verbatimize.** Upgrade transcripts you already have: given audio plus a
  trusted clean transcript, the model reproduces your content word-for-word
  and inserts only the disfluencies and vocal events actually present in the
  audio (rare-word recall jumps from 6.8% to 96.1% vs. re-transcribing).
  This turns the world's abundant clean corpora into verbatim ones, ready
  for TTS data, clinical speech analysis, and dataset construction.
- **Multilingual.** Verbatim and intended modes work across most languages
  Whisper supports. CrisperWhisper 2.0 tops the
  [Nyra Verbatim Speech Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark)
  leaderboard for disfluency F1 across ten languages, ahead of every
  closed-source alternative we tested.
- **Seamless longform.** Audio of any length, transcribed without the usual
  chunk-boundary artifacts: each window continues from the words already
  transcribed (*conditional continuation*), so there are no duplicated or
  dropped words at the seams and no fragile timestamp-token bookkeeping.
- **Production inference.** A CTranslate2 runtime with speculative decoding
  and built-in mitigation of Whisper's looping-hallucination failure mode.

## Performance

The [Nyra Verbatim Speech Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark)
scores fillers, repetitions, cut-offs, and vocal sounds as separate, typed
metrics. Its headline number is **disfluency F1**: how reliably a system
writes down the disfluencies that were actually spoken, without inventing
ones that weren't. Averaged over ten languages:

| # | System | Disfluency F1 |
|--:|--------|--------------:|
| 1 | **CrisperWhisper 2.0 Pro** | **93.5** |
| 2 | **CrisperWhisper 2.0** | **87.8** |
| 3 | ElevenLabs Scribe v2 | 79.2 |
| 4 | Microsoft MAI-Transcribe-1.5 | 77.5 |
| 5 | CrisperWhisper 1.0\* | 64.8 |
| 6 | Inworld STT | 59.5 |
| 7 | xAI Grok Speech-to-Text | 42.8 |
| 8 | Deepgram Nova-3 | 37.8 |
| 9 | Fish Audio ASR | 35.0 |
| 10 | AssemblyAI Universal-3 Pro | 30.5 |

<sub>\* CrisperWhisper 1.0 is English/German-only; its average covers those
two languages. English and German use human-labeled evaluation sets; the
other eight languages use synthetic verbatim sets. Per-language breakdowns
and how the metric is computed are in the
[benchmark post](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark).</sub>

### Word-timing accuracy

Mean absolute word-boundary error on read speech (TIMIT), lower is better:

| # | System | Boundary error |
|--:|--------|---------------:|
| 1 | **CrisperWhisper 2.0** | **29.6 ms** |
| 2 | xAI Grok Speech-to-Text | 37.1 ms |
| 3 | CTC-seg | 49.3 ms |
| 4 | ElevenLabs Scribe v2 | 51.3 ms |
| 5 | NeMo-FA | 60.0 ms |
| 6 | Deepgram Nova-3 | 63.3 ms |
| 7 | WhisperX | 64.8 ms |
| 8 | Cartesia Ink-Whisper | 69.4 ms |
| 9 | Canary | 85.5 ms |

<sub>Scored on exactly the words each system gets right. How the timings
are extracted from supervised cross-attention, plus results on
conversational speech, are in the
[aligner post](https://www.nyra-labs.com/research/attention-to-aligner).</sub>

## Install

```bash
# NVIDIA GPU (Linux): fastest, includes speculative decoding.
# An NVIDIA driver is all you need; CUDA libraries arrive via pip.
pip install "crisperwhisper[ct2]"

# Pure PyTorch: runs anywhere torch does (macOS, Windows, CPU)
pip install "crisperwhisper[transformers]"
```

## Quickstart

```python
from crisperwhisper import CrisperWhisperModel

model = CrisperWhisperModel()          # nyralabs/CrisperWhisper2.0_large
# or pick a size: CrisperWhisperModel("turbo")  # turbo / medium / small

# Verbatim transcription (default): every filler, repetition, stutter,
# false start, and vocal event
result = model.transcribe("meeting.wav", language="en")
print(result.text)

# Intended: the clean, readable version
clean = model.transcribe("meeting.wav", language="en", mode="intended")

# Word-level timestamps
result = model.transcribe("meeting.wav", language="en", word_timestamps=True)
for w in result.words:
    print(f"{w.start:6.2f}-{w.end:6.2f}  {w.word}")

# Verbatimize: upgrade an existing clean transcript with the
# disfluencies that are actually in the audio
result = model.verbatimize("clip.wav", "I think we should ship it Friday.")
```

Audio longer than 30 seconds is handled automatically (see
[longform](#what-else-is-in-the-box) below). The first load of a model
downloads it from HuggingFace and, on the `ct2` backend, converts it once
into a local cache.

### Models

| Shorthand | HuggingFace ID | Notes |
|-----------|----------------|-------|
| `"large"` (default) | `nyralabs/CrisperWhisper2.0_large` | Best open quality |
| `"turbo"` | `nyralabs/CrisperWhisper2.0_turbo` | Fastest, with some quality degradation; recommended as the speculative draft |
| `"medium"` | `nyralabs/CrisperWhisper2.0_medium` | Near-large quality; best tradeoff between size and quality |
| `"small"` | `nyralabs/CrisperWhisper2.0_small` | Smallest |
| `"large_pro"` / `"turbo_pro"` / `"medium_pro"` / `"small_pro"` | `nyralabs/CrisperWhisper2.0_<size>_pro` | **Pro**: our best models, with improved performance, hotword boosting, trained on additional proprietary data |

The standard models are released under a
[non-commercial research license](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md)
and are available for commercial licensing. The **Pro** models are available
under commercial license only. For both,
[get in touch](https://www.nyra-labs.com/crisperwhisper).

### Faster inference: speculative decoding (ct2)

A small draft model proposes tokens and the main model verifies them. Same
output, 1.3 to 1.4x faster:

```python
model = CrisperWhisperModel("large", draft_model="turbo")
result = model.transcribe("meeting.wav", language="en",
                          speculative_decoding=True)
```

## What else is in the box

Everything below works out of the box and is covered in depth in
[DOCS.md](DOCS.md):

| Option | What it does |
|--------|--------------|
| `mode="verbatim" / "intended"` | Choose what-was-said vs. what-was-meant per call |
| `word_timestamps=True` | Per-word start/end times from supervised cross-attention alignment |
| `hotwords=[...]` | Bias recognition toward names and rare terms (**Pro models only**) |
| `model.transcribe_dual(...)` | Verbatim **and** intended in one pass (ct2) |
| `model.verbatimize(audio, transcript)` | Insert real disfluencies into a trusted clean transcript |
| `model.forced_align(audio, text)` | Timings for a transcript you already have |
| Longform | Audio >30s transcribed seamlessly via conditional continuation, with no chunk-boundary duplicates, drops, or stitching |
| Hallucination mitigation | On by default: detects and suppresses Whisper's looping-repetition failure mode during decoding |
| `compute_type="float16" / "int8_float16"` | Quantization |

## How it works

Each mechanism has a deep-dive post:

- [Measuring verbatimness: the Nyra Verbatim Speech Benchmark](https://www.nyra-labs.com/research/nyra-verbatim-speech-benchmark).
  Typed metrics for fillers, repetitions, cut-offs, and vocal sounds instead
  of one opaque WER number.
- [How we unlocked multilingual style-controlled transcription at scale](https://www.nyra-labs.com/research/multilingual-style-control).
  Verbatim/intended control across languages.
- [Turning emergent cross-attention into a precise aligner](https://www.nyra-labs.com/research/attention-to-aligner).
  Supervising Whisper's alignment heads for word boundaries around 30 ms.
- [Longform transcription with conditional continuation](https://www.nyra-labs.com/research/longform-continuation).
  Resolving window seams with the words already transcribed instead of
  fragile timestamp tokens.
- [Closing the verbatim data gap with Verbatimize](https://www.nyra-labs.com/research/verbatimize).
  Upgrading clean corpora into verbatim ones at scale.
- [Faster inference and mitigating hallucinations](https://www.nyra-labs.com/research/killing-hallucinations).
  The CTranslate2 stack, speculative decoding, and the anti-looping decoder.

## Documentation

[DOCS.md](DOCS.md) covers the full API: backends and their trade-offs,
every `transcribe()` option, dual-mode transcription, forced alignment,
longform strategies, speculative-K tuning, hallucination-repair thresholds,
quantization, model conversion, and the result object.

## License

**The inference code in this repository is MIT-licensed** (see
[LICENSE](LICENSE)): use it freely, commercially or otherwise.
`crisperwhisper/features.py` is vendored from
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT, SYSTRAN).

**The model weights are not MIT.** They are released under the
[Nyra Health Non-Commercial Research License](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md):
free for research and other non-commercial use; any commercial use requires
a commercial license. The Pro models are available under commercial license
only. For commercial licensing of either,
[contact Nyra](https://www.nyra-labs.com/crisperwhisper).
