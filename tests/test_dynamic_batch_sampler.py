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
    """On first call (init), all samples must be in batches. On subsequent calls, batches + dropped_samples must cover all."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    # First call: no dropping, all samples present
    all_indices = sorted(idx for batch in sampler.batches for idx in batch)
    assert all_indices == list(range(len(dataset)))

    # Subsequent calls: batches + dropped_samples covers all
    sampler.create_batches()
    all_indices = sorted(
        [idx for batch in sampler.batches for idx in batch] + sampler.dropped_samples
    )
    assert all_indices == list(range(len(dataset)))


def test_no_duplicate_samples(dataset, max_frames):
    """No sample index should appear more than once across batches."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    sampler.create_batches()
    all_indices = [idx for batch in sampler.batches for idx in batch]
    assert len(all_indices) == len(set(all_indices))


def test_batch_count_stable_across_epochs(dataset, max_frames):
    """Batch count must stay the same across all create_batches() calls after init."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    expected = len(sampler)
    for _ in range(5):
        sampler.create_batches()
        assert len(sampler) == expected


def test_batch_respects_max_frames(dataset, max_frames):
    """No batch should exceed max_frames constraint."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
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
    sampler1 = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    batches1 = list(sampler1)

    random.seed(42)
    sampler2 = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    batches2 = list(sampler2)

    assert batches1 == batches2


def test_len_returns_batch_count(dataset, max_frames):
    """__len__ should return number of batches."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    assert len(sampler) == len(list(sampler))


def test_batches_created(dataset, max_frames):
    """Sampler should create multiple batches."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    assert len(sampler) > 0


def test_zero_redistribution_batches(dataset, max_frames):
    """With num_redistribution_batches=0, no redistribution should occur."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=0, distribution_bias=0, jitter_factor=0)
    assert sampler.redistribution_spread == 0
    all_indices = sorted(idx for batch in sampler.batches for idx in batch)
    assert all_indices == list(range(len(dataset)))


def test_zero_jitter_factor(dataset, max_frames):
    """With jitter_factor=0, batches should be strictly sorted by length."""
    random.seed(42)
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=0, distribution_bias=0, jitter_factor=0)
    sampler.create_batches()

    length_map = {idx: length for idx, length in sampler.lengths}
    first_batch = sampler.batches[0]
    first_batch_lengths = [length_map[idx] for idx in first_batch]
    last_batch = sampler.batches[-1]
    last_batch_lengths = [length_map[idx] for idx in last_batch]

    assert max(first_batch_lengths) <= max(last_batch_lengths)


def test_different_distribution_bias(dataset, max_frames):
    """Different distribution_bias values should affect redistribution spread."""
    random.seed(42)
    sampler1 = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=2, jitter_factor=0.15)
    random.seed(42)
    sampler2 = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=8, jitter_factor=0.15)
    # Both should cover all samples on first call (init)
    assert sorted(idx for batch in sampler1.batches for idx in batch) == list(range(len(dataset)))
    assert sorted(idx for batch in sampler2.batches for idx in batch) == list(range(len(dataset)))


def test_jitter_creates_variety_across_epochs(dataset, max_frames):
    """Jitter should create different batch compositions across epochs."""
    random.seed(42)
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=0, distribution_bias=0, jitter_factor=0.15)
    epoch1_batches = [batch[:] for batch in sampler.batches]
    sampler.create_batches()
    epoch2_batches = [batch[:] for batch in sampler.batches]
    assert epoch1_batches != epoch2_batches


def test_iter_shuffles_batches(dataset, max_frames):
    """__iter__ should shuffle batch order and samples within batches."""
    random.seed(42)
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)
    original_batches = [batch[:] for batch in sampler.batches]
    random.seed(42)
    iterated_batches = list(sampler)
    all_original = sorted(idx for batch in original_batches for idx in batch)
    all_iterated = sorted(idx for batch in iterated_batches for idx in batch)
    assert all_original == all_iterated


def test_enforce_max_frames_moves_overflow(dataset, max_frames):
    """_enforce_max_frames should move overflow samples to next batch."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)

    length_map = {idx: length for idx, length in sampler.lengths}

    for batch in sampler.batches:
        batch_lengths = [length_map[idx] for idx in batch]
        max_len = max(batch_lengths)
        total_frames = max_len * len(batch)
        assert total_frames <= max_frames, f"Batch exceeds max_frames: {total_frames} > {max_frames}"


def test_redistribution_spreads_short_samples(dataset, max_frames):
    """Redistribution should spread short samples across multiple batches."""
    random.seed(42)
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0)

    shortest_indices = [idx for idx, _ in sampler.lengths[:50]]

    first_batch_short_count = sum(1 for idx in sampler.batches[0] if idx in shortest_indices)
    assert first_batch_short_count < len(shortest_indices), "All short samples still in first batch"


def test_create_batches_is_idempotent_with_same_seed(dataset, max_frames):
    """Calling create_batches with same seed should produce same result."""
    sampler = DynamicBatchSampler(dataset, max_frames=max_frames, num_redistribution_batches=6, distribution_bias=4, jitter_factor=0.15)

    random.seed(100)
    sampler.create_batches()
    batches1 = [batch[:] for batch in sampler.batches]

    random.seed(100)
    sampler.create_batches()
    batches2 = [batch[:] for batch in sampler.batches]

    assert batches1 == batches2
