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
        from matcha.models.components.text_encoder import LayerNorm as ConvLayerNorm

        no_decay_modules = (
            torch.nn.Embedding,
            torch.nn.LayerNorm,
            ConvLayerNorm,
        )

        no_decay_names = set()
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if isinstance(module, no_decay_modules) or param_name == "bias":
                    no_decay_names.add(full_name)

        log.debug("No-decay params: %s", sorted(no_decay_names))

        decay_params, no_decay_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return self.hparams.optimizer(params=[
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ])

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        y_fine, y_fine_lengths = batch["y_fine"], batch["y_fine_lengths"]
        spks = batch["spks"]

        # self(...) will invoke the __call__ method from the super class, 
        # which, in its turn, invokes the forward method from matcha_tts.py
        diff_loss, dur_loss, prior_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            y_fine=y_fine,
            y_fine_lengths=y_fine_lengths,
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
        emb_keys = [k for k in ("speaker_embeddings.weight",) if
                    k in state_dict]
        if emb_keys:
            old_n_spks = state_dict[emb_keys[0]].shape[0]
            new_n_spks = self.hparams.n_spks

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
        self.batch_idx = batch_idx
        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss

        metrics = {
            f"loss/train_epoch": total_loss,
            f"sub_loss/train_diff_epoch": diff_loss,
            f"sub_loss/train_dur_epoch": dur_loss,
            f"sub_loss/train_prior_epoch": prior_loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=bs)

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss

        metrics = {
            f"loss/val_epoch": total_loss,
            f"sub_loss/val_diff_epoch": diff_loss,
            f"sub_loss/val_dur_epoch": dur_loss,
            f"sub_loss/val_prior_epoch": prior_loss,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, batch_size=bs)

        return total_loss

    # def on_before_optimizer_step(self, optimizer):
    #     # Param and Grad norm computation is rather slow, so enable it only if you must see the charts in Tensorboard.
    #     submodules = {
    #         "speaker_embeddings": self.speaker_embeddings,
    #         "encoder":            self.encoder,
    #         "decoder":            self.decoder,
    #         "phoneme_embeddings": self.encoder.emb,
    #         "enc_prenet":         self.encoder.prenet,
    #         "enc_transformer":    self.encoder.encoder._orig_mod,
    #         "enc_proj_m":         self.encoder.proj_m,
    #     }
    #     for name, module in submodules.items():
    #         # Param norm helps me check if the weight decay value from Adam / AdamW is too large.
    #         # If param_norm stays flat or slightly increases: weight decay is just fine.
    #         # If param_norm is slowly sinking: weight decay is too big; it's slowly "erasing" the model.
    #         #
    #         # It also reveals which regularizer is doing the work when both weight decay and dropout are active.
    #         # If param_norm grows freely while overfitting stays under control, it means dropout is the dominant regularizer.
    #         param_norms = torch.stack([p.detach().norm() for p in module.parameters()])
    #         self.log(f"param_norm/{name}", torch.linalg.vector_norm(param_norms), on_step=False, on_epoch=True, logger=True, batch_size=1)
    # 
    #         params_with_grad = [p for p in module.parameters() if p.grad is not None]
    #         if params_with_grad:
    #             grad_norms = torch.stack([p.grad.norm() for p in params_with_grad])
    #             self.log(f"grad_norm/{name}", torch.linalg.vector_norm(grad_norms), on_step=False, on_epoch=True, logger=True, batch_size=1)

