"""Tests for the exponential moving average (EMA) of the Decoder weights.

Training keeps the EMA next to the live Decoder and saves both in the checkpoint. Inference must be able to
load either set into the same Decoder slot, validation must compare the two on identical random draws, and
checkpoints saved with and without EMA must resume in both configurations.
"""
from types import SimpleNamespace

import pytest
import torch

from matcha.models.components.decoder_ema import (
    EMA_DECODER_PREFIX,
    LIVE_DECODER_PREFIX,
    add_ema_weights_copied_from_live,
    effective_decay,
    has_ema_weights,
    paired_flow_matching_losses,
    select_decoder_weights,
    update_ema_weights,
)
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import BASECFM

ENCODER_KEY = "encoder._orig_mod.emb.weight"
DECODER_KEYS = ("proj.weight", "proj.bias")


def make_state_dict(with_ema):
    state_dict = {ENCODER_KEY: torch.zeros(2, 2)}
    for name in DECODER_KEYS:
        state_dict[LIVE_DECODER_PREFIX + name] = torch.ones(3)
        if with_ema:
            state_dict[EMA_DECODER_PREFIX + name] = torch.full((3,), 7.0)
    return state_dict


def test_the_update_moves_each_weight_by_one_minus_decay_towards_the_live_weight():
    ema_weights = [torch.zeros(4), torch.full((2,), 10.0)]
    live_weights = [torch.ones(4), torch.zeros(2)]

    update_ema_weights(ema_weights, live_weights, decay=0.9)

    torch.testing.assert_close(ema_weights[0], torch.full((4,), 0.1))
    torch.testing.assert_close(ema_weights[1], torch.full((2,), 9.0))


def test_the_decay_ramps_up_from_the_first_step_and_never_exceeds_the_configured_value():
    first_step_decay = effective_decay(0.9999, step=0)
    resumed_run_decay = effective_decay(0.9999, step=2_500_000)

    assert first_step_decay == pytest.approx(0.1)
    assert resumed_run_decay == 0.9999


def test_inference_loads_the_ema_weights_into_the_live_decoder_slot():
    state_dict, uses_ema = select_decoder_weights(make_state_dict(with_ema=True), use_decoder_ema=True)

    assert uses_ema
    assert not has_ema_weights(state_dict)
    for name in DECODER_KEYS:
        torch.testing.assert_close(state_dict[LIVE_DECODER_PREFIX + name], torch.full((3,), 7.0))
    assert ENCODER_KEY in state_dict


def test_inference_can_ask_for_the_live_weights_of_a_checkpoint_with_ema():
    state_dict, uses_ema = select_decoder_weights(make_state_dict(with_ema=True), use_decoder_ema=False)

    assert not uses_ema
    assert not has_ema_weights(state_dict)
    torch.testing.assert_close(state_dict[LIVE_DECODER_PREFIX + "proj.bias"], torch.ones(3))


def test_a_checkpoint_without_ema_falls_back_to_the_live_weights():
    original = make_state_dict(with_ema=False)

    state_dict, uses_ema = select_decoder_weights(original, use_decoder_ema=True)

    assert not uses_ema
    assert state_dict.keys() == original.keys()


def test_an_ema_weight_without_a_live_counterpart_is_rejected():
    state_dict = make_state_dict(with_ema=True)
    state_dict[EMA_DECODER_PREFIX + "removed_layer.weight"] = torch.zeros(1)

    with pytest.raises(KeyError, match="no live Decoder counterpart"):
        select_decoder_weights(state_dict, use_decoder_ema=True)


def test_an_ema_seeded_from_live_weights_is_an_independent_copy():
    state_dict = make_state_dict(with_ema=False)

    add_ema_weights_copied_from_live(state_dict)
    state_dict[LIVE_DECODER_PREFIX + "proj.bias"].add_(1.0)

    torch.testing.assert_close(state_dict[EMA_DECODER_PREFIX + "proj.bias"], torch.ones(3))
    assert state_dict.keys() == make_state_dict(with_ema=True).keys()


@pytest.mark.parametrize("model_uses_ema, checkpoint_has_ema", [(True, False), (False, True), (True, True), (False, False)])
def test_a_resumed_checkpoint_holds_ema_weights_only_when_the_model_keeps_an_ema(model_uses_ema, checkpoint_has_ema):
    """Lightning loads the state_dict strictly, so any mismatch here would stop the resume."""
    if model_uses_ema:
        model = SimpleNamespace(decoder_ema=torch.nn.Module())
    else:
        model = SimpleNamespace(decoder_ema=None)
    state_dict = make_state_dict(with_ema=checkpoint_has_ema)

    BaseLightningClass.match_checkpoint_ema_weights_to_model(model, state_dict)

    assert has_ema_weights(state_dict) == model_uses_ema
    assert ENCODER_KEY in state_dict


class NoiseRecordingEstimator(torch.nn.Module):
    """Stands in for a Decoder whose output is its single weight, and keeps the inputs it was called with."""

    def __init__(self, weight_value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight_value))
        self.calls = []

    def forward(self, y, mask, mu, t):
        self.calls.append((y.clone(), t.clone(), self.weight.detach().clone()))
        return torch.zeros_like(y) + self.weight


def make_paired_reading():
    cfm = BASECFM(n_feats=4, cfm_params=SimpleNamespace(solver="euler", sigma_min=1e-4, use_mu_prior=False))
    cfm.estimator = NoiseRecordingEstimator(weight_value=0.0)
    ema_weights = [torch.tensor(0.5)]
    x1 = torch.randn(3, 4, 8)
    mu = torch.randn(3, 4, 8)
    mask = torch.ones(3, 1, 8)
    live_losses, ema_losses = paired_flow_matching_losses(cfm, ema_weights, x1, mask, mu)
    return cfm, ema_weights, live_losses, ema_losses


def test_the_ema_readings_replay_the_timesteps_and_noise_of_the_live_readings():
    cfm, _, live_losses, ema_losses = make_paired_reading()

    live_calls = cfm.estimator.calls[:2]
    ema_calls = cfm.estimator.calls[2:]
    for (live_input, live_t, _), (ema_input, ema_t, _) in zip(live_calls, ema_calls):
        torch.testing.assert_close(live_input, ema_input)
        torch.testing.assert_close(live_t, ema_t)
    assert live_losses != ema_losses


def test_the_ema_readings_run_on_the_ema_weights_and_the_live_weights_come_back():
    cfm, ema_weights, _, _ = make_paired_reading()

    weights_seen = [weight.item() for _, _, weight in cfm.estimator.calls]

    assert weights_seen == [0.0, 0.0, 0.5, 0.5]
    assert cfm.estimator.weight.item() == 0.0
    assert ema_weights[0].item() == 0.5
