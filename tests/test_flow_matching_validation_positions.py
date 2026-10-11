"""Tests for the timestep that the flow matching loss is computed at.

Training draws a random t for each sample. Validation adds a second reading pinned near the end of the
noise-to-mel trajectory (LATE_TRAJECTORY_T), where the decoder's velocity error is largest and the average
over random timesteps hides it. That reading feeds sub_loss/val_diff_late_epoch.
"""
from types import SimpleNamespace

import torch

from matcha.models.components.flow_matching import (
    BASECFM,
    LATE_TRAJECTORY_T,
)

BATCH_SIZE = 3
N_FEATS = 4
N_FRAMES = 8


class TimestepRecordingEstimator(torch.nn.Module):
    """Stands in for the Decoder and keeps every t it was called with, so the test can inspect them."""

    def __init__(self):
        super().__init__()
        self.recorded_timesteps = []

    def forward(self, y, mask, mu, t):
        self.recorded_timesteps.append(t.detach().clone())
        return torch.zeros_like(y)


def make_cfm():
    cfm_params = SimpleNamespace(solver="euler", sigma_min=1e-4, use_mu_prior=False)
    cfm = BASECFM(n_feats=N_FEATS, cfm_params=cfm_params)
    cfm.estimator = TimestepRecordingEstimator()
    return cfm


def compute_loss_once(cfm, sample_late_trajectory=False):
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)
    return cfm.compute_loss(x1=x1, mask=mask, mu=mu, sample_late_trajectory=sample_late_trajectory)


def test_training_path_still_draws_random_timesteps():
    """Without the late reading the timestep must still be drawn, otherwise training would collapse to one point."""
    cfm = make_cfm()
    for _ in range(8):
        compute_loss_once(cfm)

    all_drawn = torch.cat([t.reshape(-1) for t in cfm.estimator.recorded_timesteps])
    assert all_drawn.min() != all_drawn.max()
    assert all_drawn.min() >= 0.0
    assert all_drawn.max() <= 1.0


def test_late_reading_pins_every_sample_to_the_late_position():
    cfm = make_cfm()
    compute_loss_once(cfm, sample_late_trajectory=True)

    (recorded,) = cfm.estimator.recorded_timesteps
    assert recorded.shape == (BATCH_SIZE,)
    assert torch.allclose(recorded, torch.full_like(recorded, LATE_TRAJECTORY_T))
