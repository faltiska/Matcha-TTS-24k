"""
Precompute phoneme durations using facebook/wav2vec2-lv-60-espeak-cv-ft forced alignment.

## What training needs
The duration predictor is trained to predict, for each symbol in the eSpeak phoneme string
(IPA phonemes, stress markers, spaces, punctuation), how many mel frames that symbol spans.
The sum of all durations must equal the total number of mel frames in the utterance.

## What wav2vec2 + forced alignment offers
wav2vec2-lv-60-espeak-cv-ft is an ASR model whose vocabulary is a set of IPA phonemes.
Given audio, it produces per-frame log-probabilities over that vocabulary.
forced_align (CTC forced alignment) takes those log-probs and a target phoneme sequence and
returns a frame-level label sequence: each frame is assigned either a phoneme token or a blank.
The blank is the CTC blank token, not silence — it separates repeated tokens in CTC.

Two mismatches with what training needs:
1. The w2v2 vocab contains only IPA phonemes. Stress markers (ˈˌ), length marks (ː), spaces,
   and punctuation are not in the vocab, so forced_align cannot assign frames to them directly.
2. Frames not assigned to any phoneme come back as blank. These blank frames between two
   consecutive phonemes represent the actual acoustic gap (pause, silence, transition) between
   them — information that must not be discarded.

## How we infer what training needs
For each IPA phoneme in the eSpeak string, we count the frames assigned to it by forced_align
and convert to mel frames. This gives the phoneme durations directly.

For the unknown symbols between phonemes (spaces, stress markers, punctuation), we collect the
blank frames that fall in the gap between the preceding and following phoneme. Those blank frames
represent the acoustic pause in that position. We distribute them evenly across the unknown
symbols in that slot. In practice there is usually only one space between words, so it receives
the full gap duration — capturing real inter-word pauses.

Output CSV format (one file per input CSV, e.g. train.dur.csv):
  rel_base|spk|phonemes|durations
  0/0001|0|dʒˈæk hˈɒləweɪ...|2 3 1 4 ...

Durations are in mel frames (target_sr / hop_length).

Usage:
  python -m matcha.utils.precompute_durations -i configs/data/corpus-24k.yaml
  python -m matcha.utils.precompute_durations -i configs/data/corpus-24k.yaml --model facebook/wav2vec2-lv-60-espeak-cv-ft
"""

import os
import sys
import logging
logging.basicConfig()
for _name in ("phonemizer", "phonemizer.backend", "phonemizer.backend.espeak", "phonemizer.backend.espeak.espeak"):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.ERROR)
    _log.propagate = False
import argparse
from pathlib import Path

# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

import unicodedata
import torch
import torchaudio
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from matcha.text.phonemizers import multilingual_phonemizer
from matcha.text.symbols import symbols as matcha_symbols
from matcha.utils.precompute_corpus import _load_yaml_config, _resolve_path, parse_filelist

W2V2_SR = 16000
W2V2_HOP = 320  # samples per w2v2 frame at 16kHz = 20ms


def load_model(model_path, device):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device).eval()
    return processor, model


def _get_log_probs(wav_path, processor, model, device):
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != W2V2_SR:
        waveform = torchaudio.functional.resample(waveform, sr, W2V2_SR)
    input_values = processor(
        waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=W2V2_SR
    ).input_values.to(device)
    with torch.no_grad():
        return torch.nn.functional.log_softmax(model(input_values).logits, dim=-1)  # (1, T, vocab)


_CER_STRIP = None  # built lazily from w2v2 vocab in build_cer_strip()

def build_cer_strip(processor):
    """Strip chars that are in our symbol set but not in the w2v2 vocab (stress, punctuation, etc)."""
    global _CER_STRIP
    vocab = set(processor.tokenizer.get_vocab().keys())
    # Protect only combining/modifier codepoints that appear inside multi-codepoint IPA tokens
    # (e.g. combining tilde U+0303 in ɑ̃). Regular ASCII chars like '.' must not be protected
    # even if they appear in tokens like 's.' or 'ts.' in the w2v2 vocab.
    protected = set(c for token in vocab if len(token) > 1 for c in token if unicodedata.combining(c) or unicodedata.category(c).startswith('M'))
    strip_chars = "".join(c for c in matcha_symbols if c not in vocab and c not in protected)
    _CER_STRIP = str.maketrans("", "", strip_chars)

def cer(a, b, label=None):
    """Character error rate: edit distance over IPA chars, ignoring stress/spaces/punctuation.
    If label is given and the stripped strings have different lengths, prints both for inspection
    (length difference indicates dropped/inserted phonemes, which corrupt forced alignment)."""
    a, b = a.translate(_CER_STRIP).replace(" ", ""), b.translate(_CER_STRIP).replace(" ", "")
    if label is not None and len(a) != len(b):
        print(f"\n  [{label}] espeak: {b}\n  [{label}]   w2v2: {a}")
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n] / max(n, 1)


def decode(log_probs, processor):
    """Greedy CTC decode -> phoneme string."""
    ids = log_probs.squeeze(0).argmax(-1).tolist()
    blank = processor.tokenizer.pad_token_id
    # collapse repeats and remove blanks
    collapsed = [ids[0]] + [ids[i] for i in range(1, len(ids)) if ids[i] != ids[i-1]]
    return processor.tokenizer.decode([t for t in collapsed if t != blank])


def align(log_probs, phonemes, processor, device, target_sr, hop_length):
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id

    ctc_dim = log_probs.shape[-1]
    symbols = list(phonemes)
    known_mask = [p in vocab and vocab[p] < ctc_dim for p in symbols]
    known_ids = torch.tensor([vocab[p] for p, k in zip(symbols, known_mask) if k], device=device)

    if len(known_ids) == 0:
        return [0] * len(symbols)

    labels, _ = F.forced_align(log_probs, known_ids.unsqueeze(0), blank=blank_id)
    token_indices = labels.squeeze(0).cpu().tolist()

    # Walk alignment in order, counting frames per known position and blanks between positions
    n_known = len(known_ids)
    w2v2_counts = [0] * n_known   # phoneme frame counts
    gap_counts = [0] * (n_known + 1)  # blank frames: before pos 0, between each pair, after last
    pos = -1  # -1 = before first phoneme
    known_id_list = known_ids.cpu().tolist()
    for t in token_indices:
        if t == blank_id:
            gap_counts[pos + 1] += 1
        else:
            new_pos = pos
            while new_pos < n_known - 1 and t == known_id_list[new_pos + 1]:
                new_pos += 1
            pos = new_pos
            w2v2_counts[pos] += 1

    def to_mel(frames):
        return round(frames * W2V2_HOP * target_sr / (W2V2_SR * hop_length))

    mel_counts = [to_mel(c) for c in w2v2_counts]
    mel_gaps = [to_mel(c) for c in gap_counts]

    # Assign gap frames to the unknown symbols (spaces, punctuation) between known phonemes.
    # Each gap is distributed evenly across the unknown symbols in that slot.
    # gap slot k = blanks between known[k-1] and known[k] (slot 0 = before first, slot n = after last)
    known_positions = [i for i, k in enumerate(known_mask) if k]  # indices into symbols[]
    gap_slot = 0  # which gap we're filling
    unknown_in_slot = []  # indices of unknown symbols in current gap slot

    durations = [0] * len(symbols)
    ki = 0  # index into known_positions
    for i, (sym, known) in enumerate(zip(symbols, known_mask)):
        if known:
            durations[i] = mel_counts[ki]
            # flush previous gap slot
            if unknown_in_slot:
                share = mel_gaps[gap_slot] // len(unknown_in_slot)
                remainder = mel_gaps[gap_slot] % len(unknown_in_slot)
                for j, ui in enumerate(unknown_in_slot):
                    durations[ui] = share + (1 if j < remainder else 0)
            gap_slot = ki + 1
            unknown_in_slot = []
            ki += 1
        else:
            unknown_in_slot.append(i)
    # flush trailing gap
    if unknown_in_slot:
        share = mel_gaps[gap_slot] // len(unknown_in_slot)
        remainder = mel_gaps[gap_slot] % len(unknown_in_slot)
        for j, ui in enumerate(unknown_in_slot):
            durations[ui] = share + (1 if j < remainder else 0)

    return durations


def process_filelist(filelist_path, wav_dir, out_path, processor, model, device, target_sr, hop_length):
    entries = parse_filelist(filelist_path)
    written = skipped = 0
    total_cer = 0.0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, parts in enumerate(entries, 1):
            rel_base, spk, language, text = parts[0], parts[1], parts[2], parts[3]
            wav_path = (wav_dir / (rel_base + ".wav")).resolve()

            try:
                phonemes = multilingual_phonemizer(text, language)
                log_probs = _get_log_probs(wav_path, processor, model, device)

                asr = decode(log_probs, processor).replace(" ", "")
                sample_cer = cer(asr, phonemes, label=rel_base)
                total_cer += sample_cer
                durations = align(log_probs, phonemes, processor, device, target_sr, hop_length)
                # Enforce strictly monotonic positions so no phoneme has zero duration,
                # same logic as inference.py
                positions = [0] * len(durations)
                cumsum = 0
                for j, d in enumerate(durations):
                    cumsum += d
                    positions[j] = max(cumsum, j + 1)
                durations = [positions[j] - (positions[j-1] if j > 0 else 0) for j in range(len(positions))]
                dur_str = " ".join(str(d) for d in durations)
                out_f.write(f"{rel_base}|{spk}|{phonemes}|{dur_str}\n")
                written += 1
            except Exception as e:
                print(f"\n[compute_durations] SKIP {rel_base}: {e}")
                skipped += 1

            avg_cer = total_cer / max(written, 1)
            print(f"\r[compute_durations] {i}/{len(entries)} | {rel_base} | avg CER: {avg_cer:.3f}", end="", flush=True, file=sys.stderr)

    print(f"\n[compute_durations] {out_path.name}: written={written}, skipped={skipped}, avg CER={total_cer/max(written,1):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-config", required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    args = parser.parse_args()

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    target_sr = int(cfg["sample_rate"])
    hop_length = int(cfg["hop_length"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute_durations] Loading model {args.model} on {device}...")
    processor, model = load_model(args.model, device)
    build_cer_strip(processor)

    ctc_dim = model.config.vocab_size
    vocab = processor.tokenizer.get_vocab()
    unknown = sorted(s for s, i in vocab.items() if i >= ctc_dim)
    print(f"[compute_durations] {len(unknown)} symbols outside model vocab (will get duration 0): {unknown}")

    for filelist in [train_filelist, valid_filelist]:
        wav_dir = filelist.parent / "wav"
        out_path = filelist.with_suffix(".dur.csv")
        print(f"[compute_durations] Processing {filelist.name} -> {out_path.name}")
        process_filelist(filelist, wav_dir, out_path, processor, model, device, target_sr, hop_length)


if __name__ == "__main__":
    main()
