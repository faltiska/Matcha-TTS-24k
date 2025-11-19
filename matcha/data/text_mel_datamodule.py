import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        use_f0: bool = True,
        f0_fmin: float = 50.0,
        f0_fmax: float = 1100.0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.use_f0,
            self.hparams.f0_fmin,
            self.hparams.f0_fmax,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.use_f0,
            self.hparams.f0_fmin,
            self.hparams.f0_fmax,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        use_f0=True,
        f0_fmin=50.0,
        f0_fmax=1100.0,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.use_f0 = use_f0
        self.f0_fmin = f0_fmin
        self.f0_fmax = f0_fmax

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        sample = {
            "x": text,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": cleaned_text,
            "durations": durations,
        }
        if self.use_f0:
            f0 = self.get_f0(filepath, expected_len=mel.shape[-1])
            f0_mask = (f0 > 0).float()
            sample["f0"] = f0
            sample["f0_mask"] = f0_mask

        return sample

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_f0(self, filepath, expected_len):
        """
        Estimate F0 using torchaudio YIN and align to mel length.

        Returns:
            torch.Tensor: shape (1, expected_len)
        """
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        # Ensure mono
        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio[:1, :]

        frame_time = self.hop_length / float(self.sample_rate)
        # choose a small odd median smoothing window (in frames) to avoid exceeding available frames
        win_len = int(min(5, max(1, int(expected_len))))
        if win_len % 2 == 0:
            win_len = max(1, win_len - 1)
        # detect_pitch_frequency returns (channels, frames)
        f0 = ta.functional.detect_pitch_frequency(
            audio,
            self.sample_rate,
            frame_time=frame_time,
            win_length=win_len,
            freq_low=self.f0_fmin,
            freq_high=self.f0_fmax,
        )[0]

        T = f0.shape[-1]
        if T < expected_len:
            pad = torch.zeros(expected_len - T, dtype=f0.dtype)
            f0 = torch.cat([f0, pad], dim=-1)
        elif T > expected_len:
            f0 = f0[:expected_len]

        return f0.unsqueeze(0)

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        f0 = torch.zeros((B, 1, y_max_length), dtype=torch.float32)
        f0_mask = torch.zeros((B, 1, y_max_length), dtype=torch.float32)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]
            if "f0" in item and item["f0"] is not None:
                f_len = y_.shape[-1]
                # item["f0"] is (1, T)
                f0[i, :, :f_len] = item["f0"]
                f0_mask[i, :, :f_len] = item["f0_mask"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
            "f0": f0,
            "f0_mask": f0_mask,
        }
