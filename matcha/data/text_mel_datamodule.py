import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Sampler

from matcha.text import to_phoneme_ids, to_phonemes
from matcha.mel.extractors import get_mel_extractor
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse
from math import ceil

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class DynamicBatchSampler(Sampler):
    """Sampler that creates batches based on total frame count rather than fixed sample count.
    
    Minimizes padding waste by sorting samples by length and grouping similar-length samples
    together. Instead of fixed batch sizes, batches are sized to fit within a memory budget
    (max_frames = max_length_in_batch × batch_size), allowing more short samples or fewer
    long samples per batch.
    
    The "frames" in this class are time steps in the mel spectrogram.
    """

    # The samples from the first NUM_REDISTRIBUTION_BATCHES will be redistributed to other batches. 
    # First batches contains very short utterances, and always bundling short utterances together
    # may lead to some bias or overfitting. 
    NUM_REDISTRIBUTION_BATCHES = 6
    # Controls how short samples get redistributed.
    # Larger numbers mean more samples go to early batches. smaller numbers allow some samples to go
    # to batches at the end too. Batches are sorted by sample length. 
    DISTRIBUTION_BIAS = 4
    # Amount of random jitter added to lengths during sorting (as fraction of length)
    JITTER_FACTOR = 0.1
    
    def __init__(self, dataset, max_frames):
        self.dataset = dataset
        self.max_frames = max_frames
        self.lengths = self._get_lengths()
        batches, num_batches_received = self._create_batches()
        self.num_batches = len(batches)
        print(f"DynamicBatchSampler: {self.num_batches} batches, First {self.NUM_REDISTRIBUTION_BATCHES} batches will be redistributed to the next {num_batches_received} batches.", file=sys.stderr)
    
    def _get_lengths(self):
        """
        Get mel frame count for each sample, from the mel files.
        Use precompute_corpus to generate the mels before training.
        """
        lengths = []
        for i in range(len(self.dataset)):
            csv_row = self.dataset.filepaths_and_text[i]
            rel_base_path = csv_row[0]
            
            mel_path = Path(self.dataset.mel_dir) / (rel_base_path + ".npy")
            arr = np.load(mel_path, mmap_mode='r')
            lengths.append((i, arr.shape[-1]))
            
        return sorted(lengths, key=lambda x: x[1])
    

    def _jittered_sort(self, lengths):
        """Sort by length with small random variations, then restore exact original lengths.
        
        Adds ±JITTER_FACTOR noise to lengths for sorting only, creating variety in batch composition
        across epochs while preserving accurate length values for frame calculations.
        """
        real_lengths = {idx: length for idx, length in lengths}
        
        noisy_lengths = []
        for idx, length in lengths:
            noise = random.uniform(-length * self.JITTER_FACTOR, length * self.JITTER_FACTOR)
            noisy_lengths.append((idx, length + noise))
        
        noisy_lengths.sort(key=lambda x: x[1])
        
        sorted_lengths = []
        for idx, _ in noisy_lengths:
            sorted_lengths.append((idx, real_lengths[idx]))
        
        return sorted_lengths
    
    def _create_batches(self):
        """Group samples into dynamic batches from provided lengths list."""
        sorted_lengths = self._jittered_sort(self.lengths)
        
        batches = []
        current_batch = []
        max_len = 0
        
        for idx, length in sorted_lengths:
            new_max_len = max(length, max_len)
            total_frames = new_max_len * (len(current_batch) + 1)
            
            if total_frames > self.max_frames:
                batches.append([idx for idx, _ in current_batch])
                current_batch = []
                max_len = 0
                
            current_batch.append((idx, length))
            max_len = max(length, max_len)
        
        if current_batch:
            batches.append([idx for idx, _ in current_batch])
        
        num_batches_received = 0
        if self.NUM_REDISTRIBUTION_BATCHES > 0:
            batches, num_batches_received = self._redistribute_short_samples(batches)
        
        batches = self._enforce_max_frames(batches)

        return batches, num_batches_received

    def _redistribute_short_samples(self, batches):
        # Skip if not enough batches
        if len(batches) <= self.NUM_REDISTRIBUTION_BATCHES:
            return batches, 0
        
        # Take out self.NUM_REDISTRIBUTION_BATCHES and flatten their content into must_redistribute
        must_redistribute = []
        for i in range(self.NUM_REDISTRIBUTION_BATCHES):
            must_redistribute.extend(batches[i])
        batches = batches[self.NUM_REDISTRIBUTION_BATCHES:]
        
        # Shuffle must_redistribute so we distribute then in a different order each time
        random.shuffle(must_redistribute)

        num_batches = len(batches)
        total_samples = len(must_redistribute)
        
        # Compute how many redistributed samples each batch should get, according to a decaying shape
        # Earlier batches get more samples, later batches get fewer.
        redistribution_shape = []
        shape_sum = 0
        for batch_idx in range(num_batches):
            ratio = (num_batches - batch_idx) / num_batches
            num_to_add = ratio ** self.DISTRIBUTION_BIAS
            shape_sum = shape_sum + num_to_add 
            redistribution_shape.append(num_to_add)
            
        # Make sure the shape does not have fewer samples than the total number we have to distribute
        scale_factor = total_samples / shape_sum
        for i in range(len(redistribution_shape)):
            redistribution_shape[i] = ceil(redistribution_shape[i] * scale_factor) 
        
        # Distribute according to the shape
        num_batches_received = 0
        for batch_idx, num_to_add in enumerate(redistribution_shape):
            can_add = min(num_to_add, len(must_redistribute))
            if can_add > 0:
                batches[batch_idx].extend(must_redistribute[:can_add])
                must_redistribute = must_redistribute[can_add:]
                num_batches_received += 1
            else:
                break

        return batches, num_batches_received
    
    def _enforce_max_frames(self, batches):
        """Enforce max_frames constraint by moving overflow samples to next batch."""
        length_map = {idx: length for idx, length in self.lengths}
        
        i = 0
        while i < len(batches):
            batch = batches[i]
            batch_lengths = [length_map[idx] for idx in batch]
            max_len = max(batch_lengths)
            total_frames = max_len * len(batch)
            
            # Move samples to next batch until we're under max_frames
            while total_frames > self.max_frames and len(batch) > 1:
                # Remove last sample
                overflow_sample = batch.pop()
                
                # Add to next batch or create new batch if this is the last one
                if i + 1 < len(batches):
                    batches[i + 1].insert(0, overflow_sample)
                else:
                    batches.append([overflow_sample])
                
                # Recalculate
                batch_lengths = [length_map[idx] for idx in batch]
                max_len = max(batch_lengths)
                total_frames = max_len * len(batch)
            
            i += 1
        
        return batches
    
    def __iter__(self):
        batches, _ = self._create_batches()
        random.shuffle(batches)
        for batch in batches:
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


"""
All corpus train.csv/validate.csv files must use the following convention:
- The first column is the base relative path of the audio file, without extension (e.g., "1/filename", not "1/filename.wav")
- Second column  should be the speaker ID
- Third should be the text
- Columns shouldbe separated with |
TextMelDataset will append '.wav' for audio files, and '.npy' for mel files.
"""
class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
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
        mel_dir: Optional[str] = None,
        persistent_workers: bool = True,
        mel_backend="vocos",
        drop_last: bool = False,
        max_frames_per_batch: Optional[int] = None,            
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
            self.hparams.mel_dir,
            self.hparams.mel_backend,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
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
            self.hparams.mel_dir,
            self.hparams.mel_backend,
        )

    def train_dataloader(self):
        if self.hparams.max_frames_per_batch:
            batch_sampler = DynamicBatchSampler(
                self.trainset,
                max_frames=self.hparams.max_frames_per_batch,
            )
            return DataLoader(
                dataset=self.trainset,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=TextMelBatchCollate(self.hparams.n_spks),
                persistent_workers=(self.hparams.persistent_workers and self.hparams.num_workers > 0),
            )
        else:
            return DataLoader(
                dataset=self.trainset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                collate_fn=TextMelBatchCollate(self.hparams.n_spks),
                persistent_workers=(self.hparams.persistent_workers and self.hparams.num_workers > 0),
                drop_last=self.hparams.drop_last,
            )

    def val_dataloader(self):
        if self.hparams.max_frames_per_batch:
            batch_sampler = DynamicBatchSampler(
                self.validset,
                max_frames=self.hparams.max_frames_per_batch,
            )
            return DataLoader(
                dataset=self.validset,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=TextMelBatchCollate(self.hparams.n_spks),
                persistent_workers=(self.hparams.persistent_workers and self.hparams.num_workers > 0),
            )
        else:
            return DataLoader(
                dataset=self.validset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=TextMelBatchCollate(self.hparams.n_spks),
                persistent_workers=(self.hparams.persistent_workers and self.hparams.num_workers > 0),
                drop_last=self.hparams.drop_last,
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
        mel_dir=None,
        mel_backend="vocos",
    ):
        self.filelist_path = Path(filelist_path)
        self.filelist_dir = self.filelist_path.parent
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.mel_dir = mel_dir
        self.mel_backend = mel_backend
        self.mel_extractor = get_mel_extractor(
            mel_backend,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, csv_row):
        if len(csv_row) < 4:
            raise Exception(f"Malformed {csv_row=}")

        # rel_base_path now includes speaker subfolder, e.g., '1/filename'
        rel_base_path = csv_row[0]
        spk = int(csv_row[1])
        language = csv_row[2]
        text = csv_row[3]

        phonemes = to_phonemes(text, language=language)
        phoneme_ids = to_phoneme_ids(phonemes)
        if self.add_blank:
            phoneme_ids = intersperse(phoneme_ids, 0)
        phoneme_ids = torch.IntTensor(phoneme_ids)
        
        mel = self.get_mel(rel_base_path)

        if self.load_durations:
            durations = self.get_durations(rel_base_path, phoneme_ids)
        else:
            durations = None

        sample = {
            "x": phoneme_ids,
            "y": mel,
            "spk": spk,
            "filepath": rel_base_path,
            "x_text": phonemes,
            "durations": durations,
        }

        return sample

    def get_durations(self, rel_base_path, text):
        dur_loc = self.filelist_dir / "durations" / (rel_base_path + ".npy")

        try:
            durs = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generated the durations"
                f" first using: python -m matcha.utils.get_durations_from_trained_model or python -m matcha.utils.precompute_corpus --extract-durations\n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, rel_base_path):
        # rel_base_path is like "1/abc"
        # Try loading cached, already-normalized mel
        if self.mel_dir is not None:
            mel_path = Path(self.mel_dir) / (rel_base_path + ".npy")
            if mel_path.exists():
                arr = np.load(mel_path).astype(np.float32)
                mel = torch.from_numpy(arr).float()
                return mel

        # Compute mel from wav file if not cached
        wav_path = self.filelist_dir / "wav" / (rel_base_path + ".wav")
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        audio, sr = ta.load(wav_path)
        assert sr == self.sample_rate
        mel = self.mel_extractor(audio).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    """Collate function that pads variable-length samples to the longest sample in each batch.
    
    Pads phoneme sequences (x) and mel spectrograms (y) with zeros to match the longest
    sample in the batch, and returns actual lengths so the model can ignore padding.
    """
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
        }


if __name__ == "__main__":
    """
    Computes batch statistics to assess the space wasted on padding.
    Usage: python -m matcha.data.text_mel_datamodule <train.csv> <max_frames> > output.csv
    Example: python -m matcha.data.text_mel_datamodule data/corpus-small-24k/train.csv 40000 > output.csv
    CSV output goes to file, summary goes to console.
    """
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m matcha.data.text_mel_datamodule <train.csv> <max_frames> > output.csv")
        print("Example: python -m matcha.data.text_mel_datamodule data/corpus-small-24k/train.csv 40000 > output.csv")
        sys.exit(1)
    
    class MockDataset:
        def __init__(self, filelist_path, mel_dir):
            self.filepaths_and_text = parse_filelist(filelist_path)
            self.mel_dir = mel_dir
        def __len__(self):
            return len(self.filepaths_and_text)
    
    filelist_path = sys.argv[1]
    max_frames = int(sys.argv[2])
    mel_dir = Path(filelist_path).parent / "mels"
    
    dataset = MockDataset(filelist_path, mel_dir)
    sampler = DynamicBatchSampler(dataset, max_frames)
    
    print("batch_idx,samples,min_len,max_len,total_frames,actual_frames,padding_pct")
    
    total_wasted = 0
    total_frames = 0
    batch_count = 0
    
    # Build length map from initial lengths before iterator modifies them
    initial_lengths = sampler._get_lengths()
    length_map = {idx: length for idx, length in initial_lengths}
    
    for batch_idx, batch in enumerate(sampler):
        lengths = [length_map[idx] for idx in batch]
        min_len = min(lengths)
        max_len = max(lengths)
        batch_total_frames = max_len * len(batch)
        batch_actual_frames = sum(lengths)
        wasted_frames = batch_total_frames - batch_actual_frames
        padding_pct = 100 * wasted_frames / batch_total_frames
        total_wasted += wasted_frames
        total_frames += batch_total_frames
        batch_count += 1
        
        print(f"{batch_idx},{len(batch)},{min_len},{max_len},{batch_total_frames},{batch_actual_frames},{padding_pct:.2f}")
        
    print(f"\nNum batches: {batch_count}, Padding waste: {100*total_wasted/total_frames:.2f}%", file=sys.stderr)
