import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler


class SpeakerMelDataset(Dataset):
    """Returns (spk_id, mel_tensor) for each sample. Mels are loaded on demand."""

    def __init__(self, filelist_path: str, mel_dir: str):
        mel_dir = Path(mel_dir)
        rows = Path(filelist_path).read_text(encoding="utf-8").strip().splitlines()
        self.samples = []
        for row in rows:
            parts = row.split("|")
            mel_path = mel_dir / (parts[0] + ".npy")
            if mel_path.exists():
                self.samples.append((int(parts[1]), mel_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spk_id, mel_path = self.samples[idx]
        mel = torch.from_numpy(np.load(mel_path).astype(np.float32))
        return spk_id, mel


class SpeakerChunkSampler(Sampler):
    """
    Groups sample indices by speaker, shuffles within each speaker, then yields
    chunks of refs_per_step indices — all from the same speaker — as one batch.
    Each batch is a list of indices that the collate_fn will load as reference mels
    for a single training step.
    """

    def __init__(self, dataset: SpeakerMelDataset, refs_per_step: int):
        self.refs_per_step = refs_per_step
        self.indices_by_speaker: dict[int, list[int]] = {}
        for idx, (spk_id, _) in enumerate(dataset.samples):
            self.indices_by_speaker.setdefault(spk_id, []).append(idx)
        self._build_batches()

    def _build_batches(self):
        self.batches = []
        for indices in self.indices_by_speaker.values():
            shuffled = indices[:]
            random.shuffle(shuffled)
            for start in range(0, len(shuffled), self.refs_per_step):
                self.batches.append(shuffled[start : start + self.refs_per_step])
        random.shuffle(self.batches)

    def __iter__(self):
        self._build_batches()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    """
    batch: list of (spk_id, mel) tuples, all from the same speaker.
    Returns (spk_id, padded_mels, lengths).
    """
    spk_id = batch[0][0]
    mels = [item[1] for item in batch]
    lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mels], batch_first=True
    ).permute(0, 2, 1)  # (N, n_feats, T_max)
    return spk_id, padded, lengths


class SpeakerMelDataModule(LightningDataModule):
    def __init__(
        self,
        train_filelist_path: str,
        mel_dir: str,
        refs_per_step: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.trainset = SpeakerMelDataset(
            self.hparams.train_filelist_path,
            self.hparams.mel_dir,
        )

    def train_dataloader(self):
        sampler = SpeakerChunkSampler(self.trainset, self.hparams.refs_per_step)
        return DataLoader(
            self.trainset,
            batch_sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn,
        )
