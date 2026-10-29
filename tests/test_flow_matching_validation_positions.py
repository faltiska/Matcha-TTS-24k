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


def test_return_prediction_gives_the_same_loss_plus_the_velocities_it_was_computed_from():
    """The Decoder diagnostics read the predicted and target velocities; they must be the ones behind the loss."""
    cfm = make_cfm()
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)

    loss, pred, u = cfm.compute_loss(x1=x1, mask=mask, mu=mu, return_prediction=True)

    assert pred.shape == u.shape == x1.shape
    assert torch.allclose(loss, torch.mean((pred - u) ** 2))


def test_huber_matches_mse_below_the_threshold_and_grows_linearly_above_it():
    cfm = make_cfm()  # the estimator predicts zeros, so the error equals the target velocity u
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)

    torch.manual_seed(0)
    mse, pred, u = cfm.compute_loss(x1=x1, mask=mask, mu=mu, return_prediction=True)
    above_every_error = (pred - u).abs().max().item() + 1.0
    torch.manual_seed(0)
    huber_wide = cfm.compute_loss(x1=x1, mask=mask, mu=mu, huber_threshold=above_every_error)
    assert torch.allclose(huber_wide, mse)

    delta = 0.5
    torch.manual_seed(0)
    huber = cfm.compute_loss(x1=x1, mask=mask, mu=mu, huber_threshold=delta)
    error = (pred - u).abs()
    expected = torch.where(error < delta, error ** 2, 2 * delta * (error - 0.5 * delta)).mean()
    assert torch.allclose(huber, expected)


def make_cfm_with_late_band(band):
    cfm_params = SimpleNamespace(solver="euler", sigma_min=1e-4, use_mu_prior=False, extra_late_timestep_band=band)
    cfm = BASECFM(n_feats=N_FEATS, cfm_params=cfm_params)
    cfm.estimator = TimestepRecordingEstimator()
    return cfm


def test_late_timestep_doubles_the_batch_and_draws_the_second_half_inside_the_band():
    band = (0.75, 0.935)
    cfm = make_cfm_with_late_band(band)
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)

    for _ in range(16):
        loss, pred, u = cfm.compute_loss(x1=x1, mask=mask, mu=mu, return_prediction=True, add_late_timestep=True)
        assert pred.shape == u.shape == (2 * BATCH_SIZE, N_FEATS, N_FRAMES)
        # Both copies target the same mel, each from its own noise.
        assert not torch.allclose(u[:BATCH_SIZE], u[BATCH_SIZE:])
        # The loss is the mean over both readings.
        assert torch.allclose(loss, torch.mean((pred - u) ** 2))

    for recorded in cfm.estimator.recorded_timesteps:
        assert recorded.shape == (2 * BATCH_SIZE,)
        late = recorded[BATCH_SIZE:]
        assert late.min() >= band[0] and late.max() <= band[1]


def test_late_timestep_is_off_without_a_band_without_the_flag_and_for_the_pinned_reading():
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)

    no_band = make_cfm_with_late_band(None)
    no_band.compute_loss(x1=x1, mask=mask, mu=mu, add_late_timestep=True)

    band_without_flag = make_cfm_with_late_band((0.75, 0.935))
    band_without_flag.compute_loss(x1=x1, mask=mask, mu=mu)

    pinned = make_cfm_with_late_band((0.75, 0.935))
    pinned.compute_loss(x1=x1, mask=mask, mu=mu, sample_late_trajectory=True, add_late_timestep=True)

    for cfm in (no_band, band_without_flag, pinned):
        (recorded,) = cfm.estimator.recorded_timesteps
        assert recorded.shape == (BATCH_SIZE,)
