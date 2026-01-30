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
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

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

    def _log_losses(self, diff_loss, dur_loss, prior_loss, total_loss, bs, prefix="train"):
        # I am passing batch_size explicitly to avoid a warning from lightning/pytorch/utilities/data.py 
        # Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 28. 
        # To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.

        self.log(f"loss/{prefix}", total_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True, batch_size=bs)
        
        self.log(f"sub_loss/{prefix}_diff", diff_loss, on_step=True, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"sub_loss/{prefix}_dur", dur_loss, on_step=True, on_epoch=True, logger=True, batch_size=bs)
        self.log(f"sub_loss/{prefix}_prior", prior_loss, on_step=True, on_epoch=True, logger=True, batch_size=bs)

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

    def training_step(self, batch: Any, batch_idx: int):
        # avoids repeated recompilation of the model caused by changing parameter sizes.
        torch._dynamo.mark_dynamic(batch["x"], 1)

        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss
        
        self.log("step", self.global_step, on_step=True, prog_bar=True, logger=True, batch_size=bs)
        self._log_losses(diff_loss, dur_loss, prior_loss, total_loss, bs, prefix="train")

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        diff_loss, dur_loss, prior_loss = self.get_losses(batch)
        bs = batch["x"].shape[0]
        total_loss = dur_loss + prior_loss + diff_loss

        self._log_losses(diff_loss, dur_loss, prior_loss, total_loss, bs, prefix="val")

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
        
        param_group = optimizer.param_groups[0]
        self.log("grad_norm/learning_rate", param_group["lr"], on_step=True, on_epoch=False, logger=True)
