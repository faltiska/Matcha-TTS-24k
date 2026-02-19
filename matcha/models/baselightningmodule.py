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

from matcha import utils
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


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
            durations=batch["durations"],
        )

        return diff_loss, dur_loss, prior_loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init
        
        # Override LR from checkpoint with config LR
        config_lr = self.hparams.optimizer.keywords.get("lr")
        if config_lr is not None:
            for opt_state in checkpoint.get("optimizer_states", []):
                for param_group in opt_state.get("param_groups", []):
                    old_lr = param_group["lr"]
                    param_group["lr"] = config_lr
                    log.info(f"Overriding checkpoint LR {old_lr} with new value from config {config_lr}")
        
        # Override weight decay from checkpoint with config weight decay
        config_weight_decay = self.hparams.optimizer.keywords.get("weight_decay")
        if config_weight_decay is not None:
            for opt_state in checkpoint.get("optimizer_states", []):
                for param_group in opt_state.get("param_groups", []):
                    old_wd = param_group.get("weight_decay", 0.0)
                    param_group["weight_decay"] = config_weight_decay
                    log.info(f"Overriding checkpoint weight decay {old_wd} with new value from config {config_weight_decay}")
        
        # Check if a new speaker was added to the corpus
        if "spk_emb.weight" in checkpoint["state_dict"]:
            old_spk_emb = checkpoint["state_dict"]["spk_emb.weight"]
            old_n_spks = old_spk_emb.shape[0]
            new_n_spks = self.n_spks
            
            if old_n_spks < new_n_spks:
                emb_dim = old_spk_emb.shape[1]
                new_spk_emb = torch.zeros(new_n_spks, emb_dim, dtype=old_spk_emb.dtype)
                new_spk_emb[:old_n_spks] = old_spk_emb
                checkpoint["state_dict"]["spk_emb.weight"] = new_spk_emb
                
                # Expand optimizer state for speaker embeddings
                for opt_state in checkpoint.get("optimizer_states", []):
                    for param_id, state in opt_state.get("state", {}).items():
                        for key in ["exp_avg", "exp_avg_sq"]:
                            if key in state and state[key].shape[0] == old_n_spks:
                                expanded = torch.zeros(new_n_spks, emb_dim, dtype=state[key].dtype)
                                expanded[:old_n_spks] = state[key]
                                state[key] = expanded
                
                log.info(f"Added {new_n_spks - old_n_spks} more speaker(s) to the model.")

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches(old_len)
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")

    def training_step(self, batch: Any, batch_idx: int):
        # avoids repeated recompilation of the model caused by changing parameter sizes.
        torch._dynamo.mark_dynamic(batch["x"], 1)

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

    def on_validation_end(self) -> None:
        if not self.hparams.plot_mel_on_validation_end:
            return
        
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            
            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
          
        # I noticed VRAM usage suddenly increased, could not explain it
        # so I thought I would clean it up periodically
        torch.cuda.empty_cache()

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