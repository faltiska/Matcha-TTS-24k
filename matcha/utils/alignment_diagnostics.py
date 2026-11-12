"""
Shared building blocks for the alignment diagnostic scripts.

They recompute, on the CPU, the alignment that training computes on the GPU: the Encoder output of a
checkpoint is matched against the precomputed fine mel of a recording with monotonic alignment search.
Running on the CPU lets the diagnostics run while training holds the GPU.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.models.components.text_encoder import TextEncoder
from matcha.text.symbols import N_VOCAB, PRE_ID, POST_ID, SPACE_ID, symbols, vowels
from matcha.utils.model import LOG_DURATION_OFFSET

# MatchaTTS compiles the Text Encoder, so its weights sit under this prefix in the checkpoint state_dict.
ENCODER_PREFIX = "encoder._orig_mod."

FINE_HOP_LENGTH = 128
SAMPLE_RATE = 24000
FINE_FRAME_SECONDS = FINE_HOP_LENGTH / SAMPLE_RATE

WORD_GAP_LABEL = "_"


class LoadedEncoder(NamedTuple):
    encoder: TextEncoder
    speaker_embeddings_enc: torch.Tensor
    speaker_embeddings_dur: torch.Tensor
    epoch: int
    hyper_parameters: dict


class Utterance(NamedTuple):
    relative_path: str
    speaker: int
    text: str
    phoneme_ids: List[int]


def load_encoder(checkpoint_path: str) -> LoadedEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint["hyper_parameters"]
    state_dict = checkpoint["state_dict"]
    encoder_config = OmegaConf.create(hyper_parameters["encoder"])

    encoder = TextEncoder(
        encoder_config.encoder_params,
        encoder_config.duration_predictor_params,
        N_VOCAB,
        hyper_parameters["spk_emb_dim"],
    )
    encoder_weights = {
        key[len(ENCODER_PREFIX):]: value for key, value in state_dict.items() if key.startswith(ENCODER_PREFIX)
    }
    encoder.load_state_dict(encoder_weights, strict=True)
    encoder.eval()

    return LoadedEncoder(
        encoder=encoder,
        speaker_embeddings_enc=state_dict["speaker_embeddings_enc.weight"],
        speaker_embeddings_dur=state_dict["speaker_embeddings_dur.weight"],
        epoch=checkpoint["epoch"],
        hyper_parameters=hyper_parameters,
    )


def parse_filelist_line(line: str) -> Utterance:
    relative_path, speaker, _language, text, phoneme_ids = line.rstrip("\n").split("|")
    return Utterance(relative_path, int(speaker), text, [int(t) for t in phoneme_ids.split()])


def read_filelist(filelist_path: Path) -> Dict[int, List[Utterance]]:
    utterances_by_speaker = defaultdict(list)
    with open(filelist_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            utterance = parse_filelist_line(line)
            utterances_by_speaker[utterance.speaker].append(utterance)
    return utterances_by_speaker


def find_utterance(filelist_paths: List[Path], relative_path: str) -> Utterance:
    for filelist_path in filelist_paths:
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(relative_path + "|"):
                    return parse_filelist_line(line)
    raise KeyError(f"{relative_path} is in none of {filelist_paths}")


def load_fine_mel(mel_dir: Path, relative_path: str) -> torch.Tensor:
    """Normalized fine mel, as training reads it. shape: (n_feats, n_frames)"""
    return torch.from_numpy(np.load(Path(mel_dir) / f"{relative_path}.fine.npy").astype(np.float32))


def run_encoder(loaded: LoadedEncoder, utterance: Utterance) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns one Encoder frame per symbol, shape (n_feats, n_symbols), and the predicted durations in fine frames,
    unrounded, shape (n_symbols,).
    """
    speaker = [utterance.speaker]
    with torch.no_grad():
        symbols_tensor = torch.tensor(utterance.phoneme_ids)[None]
        lengths = torch.tensor([len(utterance.phoneme_ids)])
        encoder_frames, log_durations, _ = loaded.encoder(
            symbols_tensor, lengths, loaded.speaker_embeddings_enc[speaker], loaded.speaker_embeddings_dur[speaker]
        )
    # Training adds the offset before taking the log, so it is subtracted back here, as inference does.
    predicted_durations = torch.exp(log_durations[0, 0]) - LOG_DURATION_OFFSET
    return encoder_frames[0], predicted_durations


def alignment_scores(encoder_frames: torch.Tensor, mel: torch.Tensor) -> np.ndarray:
    """
    How well every symbol matches every mel frame. Same score as MatchaTTS.find_alignment, which training uses.

    Args:
        encoder_frames: one Encoder frame per symbol. shape: (n_feats, n_symbols)
        mel: ground truth fine mel. shape: (n_feats, n_frames)
    Returns:
        shape: (n_symbols, n_frames)
    """
    mel_energy = -0.5 * (mel ** 2).sum(dim=0, keepdim=True)
    cross_term = encoder_frames.T @ mel
    encoder_energy = -0.5 * (encoder_frames ** 2).sum(dim=0, keepdim=True).T
    return (mel_energy + cross_term + encoder_energy).double().numpy()


def monotonic_alignment_durations(scores: np.ndarray) -> np.ndarray:
    """
    Monotonic alignment search, returning the number of frames each symbol gets.
    The forward pass is vectorised over symbols, so the Python loop only runs once per frame.
    Its output matches the GPU kernel training uses (super_monotonic_align).
    """
    n_symbols, n_frames = scores.shape
    best_path_score = np.full((n_symbols, n_frames), -np.inf)
    best_path_score[0, 0] = scores[0, 0]
    for frame in range(1, n_frames):
        stay_on_symbol = best_path_score[:, frame - 1]
        move_from_previous_symbol = np.concatenate([[-np.inf], stay_on_symbol[:-1]])
        best_path_score[:, frame] = np.maximum(stay_on_symbol, move_from_previous_symbol) + scores[:, frame]

    durations = np.zeros(n_symbols, dtype=np.int64)
    symbol = n_symbols - 1
    for frame in range(n_frames - 1, -1, -1):
        durations[symbol] += 1
        if symbol == 0:
            continue
        # A symbol that has as many frames left as symbols before it must move on, or the first symbols starve.
        must_move_to_previous_symbol = symbol == frame
        previous_symbol_scores_better = best_path_score[symbol - 1, frame - 1] >= best_path_score[symbol, frame - 1]
        if must_move_to_previous_symbol or previous_symbol_scores_better:
            symbol -= 1
    return durations


def ground_truth_alignment(loaded: LoadedEncoder, utterance: Utterance, mel: torch.Tensor) -> np.ndarray:
    encoder_frames, _ = run_encoder(loaded, utterance)
    return monotonic_alignment_durations(alignment_scores(encoder_frames, mel))


def rounded_predicted_durations(predicted_durations: torch.Tensor) -> np.ndarray:
    """Rounded and floored at one frame, as inference does."""
    return predicted_durations.round().clamp(min=1).numpy().astype(np.int64)


def symbol_label(phoneme_id: int) -> str:
    is_opening_separator = PRE_ID <= phoneme_id < POST_ID
    is_closing_separator = phoneme_id >= POST_ID
    if is_opening_separator:
        return "<" + symbols[phoneme_id - PRE_ID]
    if is_closing_separator:
        return ">" + symbols[phoneme_id - POST_ID]
    return symbols[phoneme_id]


def vowel_core_positions(phoneme_ids: List[int]) -> List[int]:
    """Positions of the phoneme symbol of every vowel; its separators sit right before and after it."""
    positions = []
    for position, phoneme_id in enumerate(phoneme_ids):
        is_phoneme_symbol = phoneme_id < PRE_ID
        has_both_separators = 0 < position < len(phoneme_ids) - 1
        if is_phoneme_symbol and has_both_separators and symbols[phoneme_id] in vowels:
            positions.append(position)
    return positions


class Segment(NamedTuple):
    start_frame: int
    frames: int
    label: str
    first_symbol: int
    last_symbol: int


def phoneme_groups(phoneme_ids: List[int], durations: np.ndarray) -> List[Segment]:
    """
    One segment per phoneme group: the opening separator, the phoneme and the closing separator are merged,
    because the model spreads a sound over all three. Stress marks, length marks, word gaps and punctuation
    stay segments of their own.
    """
    segments = []
    frame = 0
    position = 0
    while position < len(phoneme_ids):
        phoneme_id = phoneme_ids[position]
        starts_a_group = PRE_ID <= phoneme_id < POST_ID
        if starts_a_group:
            frames = int(durations[position:position + 3].sum())
            label = symbols[phoneme_ids[position + 1]]
            segments.append(Segment(frame, frames, label, position, position + 2))
            position += 3
        else:
            frames = int(durations[position])
            if phoneme_id == SPACE_ID:
                label = WORD_GAP_LABEL
            else:
                label = symbols[phoneme_id]
            segments.append(Segment(frame, frames, label, position, position))
            position += 1
        frame += frames
    return segments


def word_segments(utterance: Utterance, durations: np.ndarray) -> List[Segment]:
    """One segment per word and one per word gap. Words are taken from the text, in order."""
    words = utterance.text.split()
    segments = []
    frame = 0
    word_index = 0
    current_word_start = None
    for position, phoneme_id in enumerate(utterance.phoneme_ids):
        is_word_gap = phoneme_id == SPACE_ID
        if is_word_gap:
            if current_word_start is not None:
                segments.append(_word_segment(words, word_index, current_word_start, position - 1, durations))
                word_index += 1
                current_word_start = None
            segments.append(Segment(int(durations[:position].sum()), int(durations[position]), WORD_GAP_LABEL, position, position))
            continue
        if current_word_start is None:
            current_word_start = position
    if current_word_start is not None:
        segments.append(_word_segment(words, word_index, current_word_start, len(utterance.phoneme_ids) - 1, durations))
    return segments


def _word_segment(words, word_index, first_symbol, last_symbol, durations) -> Segment:
    if word_index < len(words):
        label = words[word_index]
    else:
        label = "?"
    start_frame = int(durations[:first_symbol].sum())
    frames = int(durations[first_symbol:last_symbol + 1].sum())
    return Segment(start_frame, frames, label, first_symbol, last_symbol)


def write_audacity_labels(path: Path, segments: List[Segment]):
    with open(path, "w", encoding="utf-8") as f:
        for segment in segments:
            start_seconds = segment.start_frame * FINE_FRAME_SECONDS
            end_seconds = (segment.start_frame + segment.frames) * FINE_FRAME_SECONDS
            f.write(f"{start_seconds:.4f}\t{end_seconds:.4f}\t{segment.label}\n")
