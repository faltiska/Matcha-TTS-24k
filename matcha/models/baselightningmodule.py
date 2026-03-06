"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict, Optional, Union, Callable

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

import logging
from matcha import utils
from matcha.utils.utils import plot_tensor

log = logging.getLogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        return self.hparams.optimizer(params=self.parameters())

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        # self(...) will invoke the __call__ method from the super class, 
        # which, in its turn, invokes the forward method from matcha_tts.py
        diff_loss, dur_loss, prior_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
        )

        return diff_loss, dur_loss, prior_loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init
        
        for key in ("lr", "weight_decay"):
            self._override_optimizer_param(checkpoint, key)

        self.add_speaker_if_needed(checkpoint)

    def _override_optimizer_param(self, checkpoint, key):
        config_val = self.hparams.optimizer.keywords.get(key)
        if config_val is None:
            return
        for opt_state in checkpoint.get("optimizer_states", []):
            for param_group in opt_state.get("param_groups", []):
                old_val = param_group.get(key)
                param_group[key] = config_val
                log.info(f"Overriding checkpoint {key} {old_val} with config value {config_val}")

    def add_speaker_if_needed(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        emb_keys = [k for k in ("encoder_speaker_embeddings.weight", "duration_speaker_embeddings.weight", "decoder_speaker_embeddings.weight") if
                    k in state_dict]
        if emb_keys:
            old_n_spks = state_dict[emb_keys[0]].shape[0]
            new_n_spks = self.n_spks

            if old_n_spks < new_n_spks:
                for emb_key in emb_keys:
                    old_spk_emb = state_dict[emb_key]
                    emb_dim = old_spk_emb.shape[1]
                    new_spk_emb = torch.zeros(new_n_spks, emb_dim, dtype=old_spk_emb.dtype)
                    new_spk_emb[:old_n_spks] = old_spk_emb
                    state_dict[emb_key] = new_spk_emb

                # Expand optimizer state for speaker embeddings.
                # Optimizer states are indexed by the parameter's position in the flat list of all parameters.
                # We find the indices of the embedding parameters, then expand those.
                all_param_names = [name for name, _ in self.named_parameters()]
                emb_param_ids = {all_param_names.index(name) for name in emb_keys if name in all_param_names}
                for opt_state in checkpoint.get("optimizer_states", []):
                    for param_id, state in opt_state.get("state", {}).items():
                        if param_id not in emb_param_ids:
                            continue
                        for key in ["exp_avg", "exp_avg_sq"]:
                            if key in state:
                                emb_dim = state[key].shape[1]
                                expanded = torch.zeros(new_n_spks, emb_dim, dtype=state[key].dtype)
                                expanded[:old_n_spks] = state[key]
                                state[key] = expanded

                log.info(f"Added {new_n_spks - old_n_spks} more speaker(s) to the model.")

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches()
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")

    def training_step(self, batch: Any, batch_idx: int):
        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss
        
        metrics = {
            f"loss/train": total_loss,
            f"sub_loss/train_diff": diff_loss,
            f"sub_loss/train_dur": dur_loss,
            f"sub_loss/train_prior": prior_loss,
            "step": self.global_step,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True, batch_size=bs)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss

        metrics = {
            f"loss/val": total_loss,
            f"sub_loss/val_diff": diff_loss,
            f"sub_loss/val_dur": dur_loss,
            f"sub_loss/val_prior": prior_loss,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True, batch_size=bs)

        return total_loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log("grad_norm/grad_2.0_norm_total", norms["grad_2.0_norm_total"], on_step=True, on_epoch=False, logger=True)

        # This helps me check if the weight decay value from Adam / AdamW is too large.
        # If param_norm stays flat or slightly increases: weight decay is just fine.
        # If param_norm is slowly sinking: weight decay is too big; it's slowly "erasing" the model.
        #
        # It also reveals which regularizer is doing the work when both weight decay and dropout are active.
        # If param_norm grows freely while overfitting stays under control, it means dropout is the dominant regularizer.
        per_param_norms = torch.stack([p.detach().norm() for p in self.parameters()])
        total_param_norm = torch.linalg.vector_norm(per_param_norms)
        self.log("grad_norm/param_norm", total_param_norm, on_step=True, on_epoch=False, logger=True)