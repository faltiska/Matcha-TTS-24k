import pytest
import torch

from matcha.utils.model import box_downsample


def test_box_downsample_averages_over_three_fine_frames():
    fine_mel = torch.tensor([[[1.0, 2.0, 4.0, 8.0, 16.0]]])

    standard_mel = box_downsample(fine_mel)

    expected = torch.tensor([[[1.0, 14.0 / 3.0, 8.0]]])
    torch.testing.assert_close(standard_mel, expected)


@pytest.mark.parametrize(("fine_length", "standard_length"), [(4, 2), (5, 3)])
def test_box_downsample_halves_even_and_odd_lengths(fine_length, standard_length):
    fine_mel = torch.randn(2, 3, fine_length)

    standard_mel = box_downsample(fine_mel)

    assert standard_mel.shape == (2, 3, standard_length)


def test_box_downsample_counts_zero_padding_at_the_boundaries():
    """avg_pool1d must keep count_include_pad behaviour, matching an [1, 1, 1] / 3 filter."""
    fine_mel = torch.full((1, 2, 5), 3.0)

    standard_mel = box_downsample(fine_mel)

    expected = torch.tensor([[[2.0, 3.0, 2.0], [2.0, 3.0, 2.0]]])
    torch.testing.assert_close(standard_mel, expected)
