import pytest
import torch

from matcha.utils.model import triangular_downsample


def test_triangular_downsample_filters_around_standard_frame_centers():
    fine_mel = torch.tensor([[[1.0, 2.0, 4.0, 8.0, 16.0]]])

    standard_mel = triangular_downsample(fine_mel)

    expected = torch.tensor([[[1.25, 4.5, 14.0]]])
    torch.testing.assert_close(standard_mel, expected)


@pytest.mark.parametrize(("fine_length", "standard_length"), [(4, 2), (5, 3)])
def test_triangular_downsample_halves_even_and_odd_lengths(fine_length, standard_length):
    fine_mel = torch.randn(2, 3, fine_length)

    standard_mel = triangular_downsample(fine_mel)

    assert standard_mel.shape == (2, 3, standard_length)


def test_triangular_downsample_replicates_boundaries():
    fine_mel = torch.full((1, 2, 5), 3.0)

    standard_mel = triangular_downsample(fine_mel)

    torch.testing.assert_close(standard_mel, torch.full((1, 2, 3), 3.0))


def test_triangular_downsample_preserves_equal_total_influence_for_interior_frames():
    fine_mel = torch.zeros(2, 1, 5)
    fine_mel[0, 0, 1] = 1.0
    fine_mel[1, 0, 2] = 1.0

    standard_mel = triangular_downsample(fine_mel)

    torch.testing.assert_close(standard_mel.sum(dim=-1), torch.tensor([[0.5], [0.5]]))


def test_triangular_downsample_propagates_expected_gradients():
    fine_mel = torch.ones(1, 1, 5, requires_grad=True)

    triangular_downsample(fine_mel).sum().backward()

    expected_gradient = torch.tensor([[[0.75, 0.5, 0.5, 0.5, 0.75]]])
    torch.testing.assert_close(fine_mel.grad, expected_gradient)
