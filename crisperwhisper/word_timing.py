"""Word-level timing extraction from CrisperWhisper cross-attention.

Given a list of generated token IDs and the corresponding cross-attention
matrix (post-softmax, shape ``[num_tokens, num_encoder_frames]``), this
module produces a list of :class:`WordTimestamp` objects via:

1. Token-into-word grouping (CrisperWhisper's tokenizer uses an explicit
   space token, so we use the same boundary rules as the training-time
   timing evaluation).
2. Per-token log-probability over encoder frames (sharpened, normalized).
3. Per-frame "blank" log-probability from the mel spectrogram energy
   (silent frames are more likely to be blanks).
4. Word-level Viterbi alignment with virtual blank states between words.

This is a self-contained port of the ``viterbi_mel_word`` algorithm from
``evaluation/timing_extractors.py``; numpy-only, no torch dependency, so
it stays inside the slim ``crisperwhisper`` package surface.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from crisperwhisper.result import WordTimestamp

FRAME_DURATION_S = 0.02


def monotonize_words(words: List[WordTimestamp]) -> None:
    """In-place: clamp each word's start so it never precedes the previous
    word's end.

    Keeps a word timeline strictly forward-going -- used at longform chunk
    seams and after forced-alignment interpolation.  All entries must carry
    real (non-``None``) timings.
    """
    for j in range(1, len(words)):
        prev_end = words[j - 1].end
        if words[j].start < prev_end:
            words[j] = WordTimestamp(
                word=words[j].word,
                start=prev_end,
                end=max(prev_end, words[j].end),
            )


# ---------------------------------------------------------------------------
# Token -> word grouping (matches evaluation/timing_extractors.py logic).
# ---------------------------------------------------------------------------

_PROMPT_TAG_PREFIXES = ("[verbatim_", "[intended_")


def _is_special_token(piece: str) -> bool:
    if piece.startswith("<|") and piece.endswith("|>"):
        return True
    if any(piece.startswith(p) for p in _PROMPT_TAG_PREFIXES):
        return True
    # CrisperWhisper prompt markers: <ctx>...<ectx>, <htx>...<ehtx>, <vtx>...<evtx>
    if piece in {"<ctx>", "<ectx>", "<htx>", "<ehtx>", "<vtx>", "<evtx>", "<sot>", "<eot>"}:
        return True
    return False


def _is_space_token(token_id: int, piece: str) -> bool:
    # CrisperWhisper uses explicit space token 220; standard Whisper merges
    # spaces into the following token.
    return token_id == 220 or piece == " "


def _starts_new_word(token_id: int, piece: str) -> bool:
    if _is_space_token(token_id, piece):
        return True
    return piece.startswith(" ")


def group_tokens_into_words(
    token_ids: List[int],
    token_pieces: List[str],
) -> Tuple[List[List[int]], List[str]]:
    """Group token indices into words, returning ``(word_token_indices, word_texts)``."""
    word_token_indices: List[List[int]] = []
    word_texts: List[str] = []

    cur_idx: List[int] = []
    cur_text = ""

    def flush() -> None:
        nonlocal cur_idx, cur_text
        if cur_text.strip() == "" or not cur_idx:
            cur_idx = []
            cur_text = ""
            return
        word_token_indices.append(cur_idx)
        word_texts.append(cur_text.strip())
        cur_idx = []
        cur_text = ""

    for i, (tid, piece) in enumerate(zip(token_ids, token_pieces)):
        if _is_special_token(piece):
            flush()
            continue
        if _is_space_token(tid, piece):
            flush()
            continue
        if _starts_new_word(tid, piece) and cur_text != "":
            flush()
        cur_idx.append(i)
        cur_text += piece

    flush()
    return word_token_indices, word_texts


# ---------------------------------------------------------------------------
# Attention/mel -> log-probabilities.
# ---------------------------------------------------------------------------

def token_logp_from_attention(
    attention: np.ndarray,
    sharpen: float = 3.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert ``[T, F]`` post-softmax attention into per-token log-probs
    over encoder frames.  Optional sharpening (``>1`` makes peaks sharper).
    """
    attn = attention.astype(np.float64)
    attn[attn < 0] = 0.0
    if sharpen is not None and sharpen != 1.0:
        attn = np.power(attn, sharpen)
    s = attn.sum(axis=1, keepdims=True)
    s = np.clip(s, eps, None)
    p = attn / s
    return np.log(p + eps).astype(np.float32)


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-resample a 1D numpy array to ``target_len`` samples."""
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == target_len:
        return x.astype(np.float32)
    if x.size <= 1:
        return np.full((target_len,), float(x[0]) if x.size == 1 else 0.0, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def blank_logp_from_mel_energy(
    mel: np.ndarray,
    target_frames: int,
    audio_frames: Optional[int] = None,
    eps: float = 1e-6,
    gamma: float = 1.0,
    penalty: float = 0.0,
) -> np.ndarray:
    """Estimate per-encoder-frame "blank" (silence) log-probability from
    the mel spectrogram.  Silent frames -> high blank prob.

    Parameters
    ----------
    mel
        Log-mel spectrogram with shape ``[n_mels, n_mel_frames]`` (mel
        runs at twice the encoder frame rate -- 10ms vs 20ms).
    target_frames
        Number of encoder frames to align to (typically ``F_enc``).
    audio_frames
        Optional clamp: only use the first ``audio_frames`` encoder
        frames worth of mel (= ``audio_frames * 2`` mel frames).  Lets
        you avoid drawing blank statistics from the silent padding
        region of a shorter-than-30s chunk.
    gamma
        Exponent on the per-frame blank probability (``blank_p **=
        gamma``) before taking the log.  ``> 1`` makes the blank fire only
        on *genuinely* silent frames -- the raw ``1 - normalized_energy``
        rates ordinary low-energy (conversational / reduced) speech as
        half-silent, so the Viterbi steals word edges and collapses words.
        Default ``1.0`` (no change).
    penalty
        Constant subtracted from the blank log-probability -- a log-prior
        against entering a blank/pause state unless the silence evidence is
        strong.  Default ``0.0`` (no change).  ``gamma=3, penalty=3`` roughly
        halves conversational-speech word-boundary MAE (Buckeye 87->46ms)
        with no change on clean read speech (TIMIT ~34ms); too large a
        penalty over-extends words.
    """
    if mel.ndim == 3:
        mel = mel[0]

    if audio_frames is not None:
        mel_frames_to_use = min(mel.shape[1], audio_frames * 2)
        mel = mel[:, :mel_frames_to_use]

    energy = mel.mean(axis=0).astype(np.float32)

    p10 = float(np.percentile(energy, 10))
    p90 = float(np.percentile(energy, 90))
    denom = max(1e-6, p90 - p10)
    energy_norm = np.clip((energy - p10) / denom, 0.0, 1.0)

    energy_norm = _resample_1d(energy_norm, target_frames)
    blank_p = np.clip(1.0 - energy_norm, 1e-4, 1.0 - 1e-4)
    if gamma != 1.0:
        blank_p = np.clip(blank_p ** gamma, 1e-4, 1.0)
    logp = np.log(blank_p + eps).astype(np.float32)
    if penalty:
        logp = (logp - penalty).astype(np.float32)
    return logp


SPACE_TOKEN_ID = 220


def blank_logp_from_space_attention(
    attention: np.ndarray,
    gen_ids: List[int],
    target_frames: int,
    sharpen: float = 3.0,
    eps: float = 1e-8,
) -> Optional[np.ndarray]:
    """Estimate per-frame "blank" (inter-word pause) log-probability from the
    cross-attention of the explicit space token(s).

    CrisperWhisper's changed tokenizer emits an explicit space token
    (id ``220``) between words.  The frames that token attends to are
    precisely the inter-word gaps, which is a cleaner pause signal than
    mel-energy silence.  Returns a ``[target_frames]`` log-prob vector, or
    ``None`` when the sequence has no explicit space tokens (so the caller
    can fall back to the mel-energy estimate).
    """
    space_idx = [
        i for i, t in enumerate(gen_ids)
        if int(t) == SPACE_TOKEN_ID and i < attention.shape[0]
    ]
    if not space_idx:
        return None

    sp = attention[space_idx, :target_frames].astype(np.float64)
    sp[sp < 0] = 0.0
    if sharpen is not None and sharpen != 1.0:
        sp = np.power(sp, sharpen)

    row_sums = sp.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, eps, None)
    sp_norm = sp / row_sums  # each space token -> distribution over frames

    # A frame is "blank" if any space token concentrates its attention there.
    blank = sp_norm.max(axis=0)
    peak = float(blank.max())
    if peak > 0:
        blank = blank / peak
    blank = np.clip(blank, 1e-4, 1.0 - 1e-4)
    return np.log(blank + eps).astype(np.float32)


# ---------------------------------------------------------------------------
# Viterbi alignment with virtual blank states.
# ---------------------------------------------------------------------------

def viterbi_align_tokens_with_blanks(
    token_logp: np.ndarray,
    blank_logp: np.ndarray,
    frame_duration: float = FRAME_DURATION_S,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Viterbi alignment with virtual blanks ``[blank0, tok0, blank1,
    tok1, ..., blankT]``.  Returns ``[(start, end), ...]`` per token.

    The forward DP is vectorised over states (the per-step state loop
    in the naive implementation is the dominant cost for long chunks);
    only the time loop remains in Python, which is unavoidable because
    each step depends on the previous one.
    """
    T, F = token_logp.shape
    if T == 0 or F == 0:
        return []
    if blank_logp.shape[0] != F:
        raise ValueError("blank_logp length must match number of frames")

    S = 2 * T + 1
    neg_inf = np.float32(-1e9)

    # Build the per-state emission matrix [S, F].  Even states are
    # virtual blanks (T+1 of them, all sharing ``blank_logp``); odd
    # states emit the corresponding token row.
    emit = np.empty((S, F), dtype=np.float32)
    emit[0::2, :] = blank_logp[np.newaxis, :]
    emit[1::2, :] = token_logp

    dp = np.full((S, F), neg_inf, dtype=np.float32)
    back = np.zeros((S, F), dtype=np.int8)
    dp[0, 0] = emit[0, 0]

    # Forward DP -- vectorised over the state axis.  At frame ``f``:
    #   stay[s] = dp[s, f-1]
    #   adv[s]  = dp[s-1, f-1]    (state 0 has no "advance from -1")
    # dp[s, f] = max(stay[s], adv[s]) + emit[s, f]
    # back[s, f] = 1 iff adv[s] > stay[s]
    adv_row = np.full((S,), neg_inf, dtype=np.float32)
    for f in range(1, F):
        prev = dp[:, f - 1]
        adv_row[1:] = prev[:-1]  # state 0 keeps -inf, can never advance
        take_adv = adv_row > prev
        best = np.where(take_adv, adv_row, prev)
        dp[:, f] = best + emit[:, f]
        back[:, f] = take_adv  # implicit cast bool -> int8 (0/1)

    end_state = int(
        np.argmax(dp[:, F - 1] + (np.arange(S, dtype=np.float32) * 1e-4))
    )
    states = np.empty((F,), dtype=np.int32)
    s = end_state
    for f in range(F - 1, -1, -1):
        states[f] = s
        if f == 0:
            break
        if back[s, f] == 1:
            s -= 1

    out: List[Tuple[Optional[float], Optional[float]]] = []
    for t in range(T):
        tok_state = 2 * t + 1
        idx = np.where(states == tok_state)[0]
        if idx.size == 0:
            out.append((None, None))
        else:
            out.append((int(idx[0]) * frame_duration, int(idx[-1]) * frame_duration))
    return out


def viterbi_align_words_with_blanks(
    token_logp: np.ndarray,
    blank_logp: np.ndarray,
    word_token_indices: List[List[int]],
    frame_duration: float = FRAME_DURATION_S,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """Word-level Viterbi alignment.

    Collapses the per-token log-probabilities for each word's tokens via
    logsumexp into a single "word emission" row, then runs the
    token-level Viterbi with virtual blanks.
    """
    T, F = token_logp.shape
    W = len(word_token_indices)
    if W == 0:
        return []

    word_logp = np.full((W, F), -1e9, dtype=np.float32)
    for w, tok_idxs in enumerate(word_token_indices):
        valid = [i for i in tok_idxs if 0 <= i < T]
        if valid:
            word_logp[w, :] = np.logaddexp.reduce(token_logp[valid, :], axis=0)

    return viterbi_align_tokens_with_blanks(word_logp, blank_logp, frame_duration)


# ---------------------------------------------------------------------------
# Post-processing: distribute short inter-word silences.
# ---------------------------------------------------------------------------

def split_interword_gaps(
    words: List[WordTimestamp],
    max_gap_s: float = 0.1,
) -> List[WordTimestamp]:
    """Split short inter-word silences evenly between the adjacent words.

    The Viterbi parks the silence/space between two words in a blank state, so
    a word's ``end`` is the onset of its last emitting frame and the gap up to
    the next word's ``start`` is left unassigned. When words are spaced tightly
    (no real pause) this under-extends ends and over-delays starts relative to
    references that make word boundaries contiguous (it shows up as a negative
    end-bias on conversational/stuttered speech).

    For every adjacent pair whose gap is ``0 < gap <= max_gap_s`` we move both
    boundaries to the gap midpoint, so the trailing/leading silence is shared
    evenly. Genuine pauses (``gap > max_gap_s``) are left untouched, and
    touching/overlapping words (``gap <= 0``) are skipped. Mutates and returns
    ``words``.
    """
    if max_gap_s <= 0 or len(words) < 2:
        return words
    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if 0.0 < gap <= max_gap_s:
            mid = a.end + gap / 2.0
            a.end = float(mid)
            b.start = float(mid)
    return words


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def extract_word_timings(
    engine,
    gen_ids: List[int],
    attention: np.ndarray,
    mel: np.ndarray,
    *,
    audio_duration_s: Optional[float] = None,
    sharpen: float = 5.0,
    frame_duration: float = FRAME_DURATION_S,
    blank_source: str = "mel",
    blank_gamma: float = 3.0,
    blank_penalty: float = 3.0,
    clip_to_audio: bool = False,
    split_gap_max_s: float = 0.1,
    keep_unplaceable: bool = False,
) -> List[WordTimestamp]:
    """Build a list of :class:`WordTimestamp`s from cross-attention.

    Parameters
    ----------
    engine
        A :class:`CT2Engine` (used only to decode token ids -> pieces).
    gen_ids
        The generated token id list.
    attention
        Cross-attention matrix of shape ``[len(gen_ids), F_enc]``
        (mean-over-heads, post-softmax).  Must be 1-to-1 with
        ``gen_ids``.
    mel
        Log-mel spectrogram of the audio chunk, shape ``[n_mels,
        n_mel_frames]`` (i.e. what ``CT2Engine._feature_extractor``
        produces before wrapping).  Used to derive per-frame blank
        probabilities.
    audio_duration_s
        Audio length in seconds. Only used when ``clip_to_audio=True``
        (see below); ignored by the default full-window aligner.
    sharpen
        Attention sharpening exponent. Default 5.0 -- empirically best for
        this verbatim timing model (TIMIT 34.7 vs 39.3 at 3.0); it pairs with
        the blank shaping below (the two must move together -- ``gamma=3,
        penalty=3`` at a softer ``sharpen=3`` over-extends words and hurts
        clean speech).
    frame_duration
        Encoder frame duration in seconds (Whisper = 0.02).
    blank_source
        Where the Viterbi "blank" (pause) probabilities come from:
        ``"mel"`` (default) uses mel-energy silence; ``"space"`` uses the
        explicit space token's cross-attention (better for tokenizers with
        a dedicated space token, e.g. the legacy CrisperWhisper model);
        ``"auto"`` uses the space token when the sequence contains explicit
        space tokens (id 220) and falls back to mel energy otherwise.  The
        default is ``"mel"`` so the v2 CTranslate2 and Transformers backends
        stay numerically identical.
    blank_gamma, blank_penalty
        Shaping applied to the ``"mel"`` blank (see
        :func:`blank_logp_from_mel_energy`): ``blank_gamma`` exponentiates the
        per-frame blank probability so only genuinely silent frames score as
        blank, and ``blank_penalty`` subtracts a constant log-prior against
        blank states. Defaults ``3.0`` / ``3.0`` -- the raw mel blank
        (``gamma=1, penalty=0``) rates ordinary low-energy conversational
        speech as half-silent and the Viterbi collapses words to their
        attention peak (Buckeye words ~26% of true length, 87ms word-boundary
        MAE); the shaped blank recovers ~78% (Buckeye 46ms) with no change on
        clean read speech (TIMIT ~34ms). Pass ``gamma=1, penalty=0`` to
        recover the raw mel blank. CT2/Transformers stay numerically identical
        (both use these defaults).
    clip_to_audio
        If ``True``, clip the attention/mel analysis to ``audio_duration_s``
        (legacy behaviour). Defaults to ``False``: the full encoder window is
        used and the Viterbi blank states absorb the silent padding, which is
        substantially more accurate for the blank-based aligner (clipping
        recomputes the mel-energy percentiles over the speech region only and
        collapses words onto their attention peak).
    split_gap_max_s
        If ``> 0``, post-process the Viterbi output with
        :func:`split_interword_gaps`: every inter-word gap up to this length
        (seconds) is split evenly between the two words (both boundaries move
        to the gap midpoint), while longer gaps -- genuine pauses -- are left
        as-is. Default ``0.1`` (100 ms): the Viterbi parks inter-word silence in
        a blank state, so word ends sit at the onset of the last emitting frame
        and tight boundaries are left non-contiguous (a negative end-bias). The
        100 ms split makes those boundaries contiguous and is a uniform win --
        MAE TIMIT 34.5->30.8, Buckeye 41.8->37.3, FluencyBank 103.3->101.1, with
        mIoU and F1@50 up everywhere -- without filling genuine pauses. Pass
        ``0.0`` to disable. CT2/Transformers stay numerically identical.
    keep_unplaceable
        If ``False`` (default), words for which Viterbi could not place a state
        are skipped, so the returned list contains only placeable words (the
        legacy behaviour). If ``True``, those words are kept as placeholders
        (``start=None``, ``end=None``) so the returned list is positionally
        **1-to-1 with the word segmentation** -- each entry's ``.word`` carries
        the word text in order. Used by the longform drop logic, which needs a
        word list aligned 1-to-1 with the text and anchors its decision only on
        the placeable ``.start`` values. The placeable words' timings are
        identical in both modes (the same ``split_interword_gaps`` pass runs on
        them).

    Returns
    -------
    A list of :class:`WordTimestamp` objects (chunk-local seconds). With
    ``keep_unplaceable=False`` only placeable words are returned; with
    ``keep_unplaceable=True`` the list is 1-to-1 with the word segmentation and
    unplaceable words appear as ``start=None``/``end=None`` placeholders.
    """
    if not gen_ids or attention.shape[0] == 0:
        return []

    # Tokens -> word groups (mirrors group_token_rows_into_words in
    # evaluation/timing_extractors.py).
    tok_pieces = [engine.tokenizer.decode([t]) for t in gen_ids]
    word_token_indices, word_texts = group_tokens_into_words(gen_ids, tok_pieces)
    if not word_token_indices:
        return []

    # Frame window for alignment. By default we use the FULL encoder window and
    # let the Viterbi blank states absorb the silent (padding) region.
    #
    # Clipping to the audio duration is a *bug* for the blank-based aligner: it
    # recomputes the mel-energy percentiles over the speech region only, so the
    # quiet frames *within* and *between* words fall below the new p10 and get
    # marked as blank. The Viterbi then collapses each word onto its attention
    # peak, biasing starts late and ends early (~138ms vs ~37ms word-boundary
    # MAE on TIMIT for the timing model). It also shrinks the token_logp
    # frame-normalisation denominator. The full window keeps the silence-padding
    # floor as the percentile reference and lets the blank states handle the
    # pad, matching evaluation/evaluate_timing_alignment.py (viterbi_blank,
    # which never clips). ``clip_to_audio=True`` restores the legacy behaviour
    # for extractors that have no blank/pause signal.
    F_enc_total = attention.shape[1]
    if clip_to_audio and audio_duration_s is not None:
        actual_frames = min(
            F_enc_total, max(1, int(round(audio_duration_s / frame_duration)))
        )
        mel_audio_frames: Optional[int] = actual_frames
    else:
        actual_frames = F_enc_total
        mel_audio_frames = None

    attn = attention[:, :actual_frames].astype(np.float32)

    # log-probabilities for token / blank emissions
    tok_logp = token_logp_from_attention(attn, sharpen=sharpen)

    blank_logp = None
    if blank_source in ("auto", "space"):
        blank_logp = blank_logp_from_space_attention(
            attn, gen_ids, target_frames=actual_frames, sharpen=sharpen,
        )
        if blank_logp is None and blank_source == "space":
            # explicitly requested but no space tokens -> fall back to mel
            pass
    if blank_logp is None:
        blank_logp = blank_logp_from_mel_energy(
            mel, target_frames=actual_frames, audio_frames=mel_audio_frames,
            gamma=blank_gamma, penalty=blank_penalty,
        )

    word_timings = viterbi_align_words_with_blanks(
        tok_logp, blank_logp, word_token_indices, frame_duration=frame_duration,
    )

    placed: List[WordTimestamp] = []
    placed_idx: List[int] = []
    for idx, (word, (start, end)) in enumerate(zip(word_texts, word_timings)):
        if start is None or end is None:
            continue
        placed.append(WordTimestamp(word=word, start=float(start), end=float(end)))
        placed_idx.append(idx)

    if split_gap_max_s and split_gap_max_s > 0:
        placed = split_interword_gaps(placed, max_gap_s=split_gap_max_s)

    if not keep_unplaceable:
        return placed

    # 1-to-1 with the word segmentation: re-insert placeholders (None timings)
    # for words the Viterbi could not place, preserving order.
    full: List[WordTimestamp] = [
        WordTimestamp(word=w, start=None, end=None) for w in word_texts
    ]
    for wt, idx in zip(placed, placed_idx):
        full[idx] = wt
    return full
