import random
import pytest
from pathlib import Path
from matcha.data.text_mel_datamodule import TextMelDataset, DynamicBatchSampler

# pytest tests/test_dynamic_batch_sampler.py --max-frames 33000

@pytest.fixture(scope="module")
def dataset():
    """Load corpus-small-24k dataset once for all tests."""
    return TextMelDataset(
        filelist_path="/home/alfred/projects/matcha-tts/data/corpus-small-24k/train.csv",
        n_spks=10,
        add_blank=True,
        n_fft=1024,
        n_mels=100,
        sample_rate=24000,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=12000,
        data_parameters={"mel_mean": -3.186903, "mel_std": 5.36849},
        seed=1234,
        load_durations=False,
        mel_dir="/home/alfred/projects/matcha-tts/data/corpus-small-24k/mels",
        mel_backend="vocos"
    )


def test_all_samples_covered(dataset, max_frames):
    """Each sample should appear exactly once per epoch."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames)
    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]
    assert sorted(all_indices) == list(range(len(dataset)))


def test_batch_respects_max_frames(dataset, max_frames):
    """No batch should exceed max_frames constraint."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames)
    batches = list(sampler)
    
    length_map = {idx: length for idx, length in sampler.lengths}
    for batch in batches:
        batch_lengths = [length_map[idx] for idx in batch]
        max_len = max(batch_lengths)
        total_frames = max_len * len(batch)
        assert total_frames <= max_frames


def test_deterministic_with_seed(dataset, max_frames):
    """Same seed should produce same batch order."""
    random.seed(42)
    sampler1 = DynamicBatchSampler(dataset, max_frames=max_frames)
    batches1 = list(sampler1)
    
    random.seed(42)
    sampler2 = DynamicBatchSampler(dataset, max_frames=max_frames)
    batches2 = list(sampler2)
    
    assert batches1 == batches2


def test_len_returns_batch_count(dataset, max_frames):
    """__len__ should return number of batches."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames)
    assert len(sampler) == len(list(sampler))


def test_batches_created(dataset, max_frames):
    """Sampler should create multiple batches."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames)
    assert len(sampler) > 0
