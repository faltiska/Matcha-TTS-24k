"""Tests for the Decoder's late-trajectory velocity readings that feed abs_error_quantiles/val_decoder_late_*.

The late reading is pinned to LATE_TRAJECTORY_T. Its predicted and target velocities must be the ones behind
sub_loss/val_diff_late_epoch, and with EMA the live and EMA readings must see the same noise.
"""
from types import SimpleNamespace

import torch

from matcha.models.components.decoder_ema import paired_flow_matching_losses, read_flow_matching_losses
from matcha.models.components.flow_matching import BASECFM, LATE_TRAJECTORY_T

BATCH_SIZE = 3
N_FEATS = 4
N_FRAMES = 8


class ScaledEstimator(torch.nn.Module):
    """Stands in for the Decoder: predicts scale * y, and records each t, so the weights change the prediction."""

    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.recorded_timesteps = []

    def forward(self, y, mask, mu, t):
        self.recorded_timesteps.append(t.detach().clone())
        return self.scale * y


def make_cfm(scale=0.5):
    cfm_params = SimpleNamespace(solver="euler", sigma_min=1e-4, use_mu_prior=False)
    cfm = BASECFM(n_feats=N_FEATS, cfm_params=cfm_params)
    cfm.estimator = ScaledEstimator(scale)
    return cfm


def make_batch():
    x1 = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mu = torch.randn(BATCH_SIZE, N_FEATS, N_FRAMES)
    mask = torch.ones(BATCH_SIZE, 1, N_FRAMES)
    mask[0, :, -2:] = 0.0
    return x1, mask, mu


def test_late_prediction_is_pinned_and_matches_the_late_loss():
    cfm = make_cfm()
    x1, mask, mu = make_batch()

    diff_loss, late_loss, late_pred, late_target = read_flow_matching_losses(cfm, x1, mask, mu)

    _, late_t = cfm.estimator.recorded_timesteps
    assert torch.allclose(late_t, torch.full_like(late_t, LATE_TRAJECTORY_T))
    assert late_pred.shape == late_target.shape == x1.shape
    full_mask = mask.expand_as(x1)
    expected = torch.sum(((late_pred - late_target) * full_mask) ** 2) / full_mask.sum()
    assert torch.allclose(late_loss, expected)
    assert diff_loss.ndim == 0


def test_live_and_ema_late_readings_share_the_noise_and_differ_only_by_the_weights():
    cfm = make_cfm(scale=0.5)
    ema_weights = [torch.tensor(0.25)]
    x1, mask, mu = make_batch()

    # Validation runs without grad; swapping weights in place on a leaf that requires grad needs it too.
    with torch.no_grad():
        live, ema = paired_flow_matching_losses(cfm, ema_weights, x1, mask, mu)

    # Same t and noise, so the late target velocities are identical.
    assert torch.equal(live[3], ema[3])
    assert not torch.allclose(live[2], ema[2])
    # The live weights are restored after the EMA readings.
    assert cfm.estimator.scale.item() == 0.5
