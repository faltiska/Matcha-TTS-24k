"""Tests for the optimizer parameter-group handling in BaseLightningClass.

configure_optimizers() deliberately excludes embeddings, normalization parameters, and biases from
weight decay. on_load_checkpoint() re-applies the configured optimizer settings to a resumed
checkpoint, and must respect that exclusion rather than flattening every group to the global value.
"""
from functools import partial

import pytest
import torch

from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.text_encoder import LayerNorm as ConvLayerNorm

CONFIGURED_LR = 0.003
CONFIGURED_WEIGHT_DECAY = 0.125

# Values a checkpoint saved before the per-group fix would contain: the global weight decay written
# over every group, including the one meant to stay at zero.
STALE_LR = 0.001
STALE_WEIGHT_DECAY = 0.0002


class ParamGroupModel(BaseLightningClass):
    """A model exercising every branch of the decay/no-decay rule.

    The two groups hold different numbers of parameters on purpose: equal sizes would let the resume
    override pair a saved group with the wrong spec group unnoticed.
    """

    def __init__(self, weight_decay=CONFIGURED_WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters({
            "optimizer": partial(torch.optim.AdamW, lr=CONFIGURED_LR, weight_decay=weight_decay),
            "n_spks": 1,
        })
        # Decay: plain weights.
        self.first_linear = torch.nn.Linear(4, 4, bias=True)
        self.second_linear = torch.nn.Linear(4, 4, bias=True)
        self.conv = torch.nn.Conv1d(4, 4, 1, bias=True)
        # No decay: embeddings, both flavours of normalization, and the biases above.
        self.embedding = torch.nn.Embedding(6, 4)
        self.layer_norm = torch.nn.LayerNorm(4)
        self.conv_layer_norm = ConvLayerNorm(4)


def make_checkpoint(model):
    """A checkpoint holding the optimizer state the model would really save."""
    return {
        "epoch": 7,
        "state_dict": model.state_dict(),
        "optimizer_states": [model.configure_optimizers().state_dict()],
    }


def saved_groups(checkpoint):
    return checkpoint["optimizer_states"][0]["param_groups"]


def group_values(checkpoint, key):
    return [group[key] for group in saved_groups(checkpoint)]


def corrupt_as_old_checkpoints_did(checkpoint):
    """Reproduce the state left by the old override, which ignored per-group settings."""
    for group in saved_groups(checkpoint):
        group["lr"] = STALE_LR
        group["weight_decay"] = STALE_WEIGHT_DECAY


def test_the_no_decay_group_holds_embeddings_normalization_and_biases():
    model = ParamGroupModel()

    decay_group, no_decay_group = model._optimizer_param_group_spec()

    assert decay_group["overrides"] == {}
    assert no_decay_group["overrides"] == {"weight_decay": 0.0}
    assert set(decay_group["names"]) == {
        "first_linear.weight",
        "second_linear.weight",
        "conv.weight",
    }
    assert set(no_decay_group["names"]) == {
        "first_linear.bias",
        "second_linear.bias",
        "conv.bias",
        "embedding.weight",
        # ConvLayerNorm names its parameters gamma/beta, so only the module-type rule catches them.
        "conv_layer_norm.gamma",
        "conv_layer_norm.beta",
        "layer_norm.weight",
        "layer_norm.bias",
    }


def test_a_fresh_optimizer_zeroes_weight_decay_only_for_the_no_decay_group():
    model = ParamGroupModel()

    optimizer = model.configure_optimizers()

    assert [group["weight_decay"] for group in optimizer.param_groups] == [CONFIGURED_WEIGHT_DECAY, 0.0]


def test_resuming_restores_the_zero_weight_decay_group_from_a_corrupted_checkpoint():
    """The regression this guards: the no-decay group silently gained weight decay on every resume."""
    model = ParamGroupModel()
    checkpoint = make_checkpoint(model)
    corrupt_as_old_checkpoints_did(checkpoint)

    model.on_load_checkpoint(checkpoint)

    assert group_values(checkpoint, "weight_decay") == [CONFIGURED_WEIGHT_DECAY, 0.0]


def test_resuming_applies_the_configured_lr_to_every_group():
    model = ParamGroupModel()
    checkpoint = make_checkpoint(model)
    corrupt_as_old_checkpoints_did(checkpoint)

    model.on_load_checkpoint(checkpoint)

    assert group_values(checkpoint, "lr") == [CONFIGURED_LR, CONFIGURED_LR]


def test_resuming_a_correct_checkpoint_changes_nothing():
    """Repeated resumes must not drift, since each resume is saved into the next checkpoint."""
    model = ParamGroupModel()
    checkpoint = make_checkpoint(model)

    model.on_load_checkpoint(checkpoint)
    after_first_resume = group_values(checkpoint, "weight_decay")
    model.on_load_checkpoint(checkpoint)

    assert after_first_resume == [CONFIGURED_WEIGHT_DECAY, 0.0]
    assert group_values(checkpoint, "weight_decay") == after_first_resume


def test_a_changed_configuration_reaches_the_decay_group_but_not_the_no_decay_group():
    checkpoint = make_checkpoint(ParamGroupModel(weight_decay=CONFIGURED_WEIGHT_DECAY))
    model_with_new_config = ParamGroupModel(weight_decay=0.5)

    model_with_new_config.on_load_checkpoint(checkpoint)

    assert group_values(checkpoint, "weight_decay") == [0.5, 0.0]


def test_the_override_follows_the_group_order_declared_by_the_spec():
    """configure_optimizers() and the resume override must not disagree about group order.

    Both read the order from _optimizer_param_group_spec(), so reversing it there has to keep the two
    in step. Before this was centralised, the override assumed the decay group came first and silently
    swapped the two values when the order changed.
    """

    class ReversedSpecModel(ParamGroupModel):
        def _optimizer_param_group_spec(self):
            return list(reversed(super()._optimizer_param_group_spec()))

    model = ReversedSpecModel()
    checkpoint = make_checkpoint(model)
    expected = [group["weight_decay"] for group in model.configure_optimizers().param_groups]
    corrupt_as_old_checkpoints_did(checkpoint)

    model.on_load_checkpoint(checkpoint)

    assert expected == [0.0, CONFIGURED_WEIGHT_DECAY]
    assert group_values(checkpoint, "weight_decay") == expected


def test_nothing_is_overridden_when_the_optimizer_config_omits_the_key():
    model = ParamGroupModel()
    checkpoint = make_checkpoint(model)
    corrupt_as_old_checkpoints_did(checkpoint)
    model.hparams.optimizer = partial(torch.optim.AdamW)

    model.on_load_checkpoint(checkpoint)

    assert group_values(checkpoint, "weight_decay") == [STALE_WEIGHT_DECAY, STALE_WEIGHT_DECAY]
    assert group_values(checkpoint, "lr") == [STALE_LR, STALE_LR]


def test_a_checkpoint_that_does_not_match_the_current_split_is_rejected():
    """torch cannot load such an optimizer state, so the resume must stop with a readable reason."""
    model = ParamGroupModel()
    checkpoint = make_checkpoint(model)
    # A checkpoint from a different architecture: fewer parameters than the current model has.
    saved_groups(checkpoint)[1]["params"] = [0, 1]

    with pytest.raises(AssertionError, match="cannot be resumed"):
        model.on_load_checkpoint(checkpoint)


def test_frozen_parameters_are_left_out_of_the_optimizer_groups():
    """Fine-tuning freezes most of the model, so the split must only cover trainable parameters."""
    model = ParamGroupModel()
    model.first_linear.weight.requires_grad = False
    model.embedding.weight.requires_grad = False

    decay_group, no_decay_group = model._optimizer_param_group_spec()

    assert "first_linear.weight" not in decay_group["names"]
    assert "embedding.weight" not in no_decay_group["names"]
