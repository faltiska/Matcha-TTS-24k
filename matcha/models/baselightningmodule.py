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

    def _optimizer_param_group_spec(self):
        """
        The optimizer's parameter groups, as names plus the per-group option overrides.

        Embeddings, normalization parameters, and biases are excluded from weight decay, as it would erase speaker identity from embeddings, or the learned scale/shift of normalization layers.

        This is the single source of truth for the decay/no-decay split.
        """
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

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if name in no_decay_names else decay).append(name)

        log.debug("No-decay params: %s", sorted(no_decay))

        return [
            {"names": decay, "overrides": {}},
            {"names": no_decay, "overrides": {"weight_decay": 0.0}},
        ]

    def configure_optimizers(self) -> Any:
        params_by_name = dict(self.named_parameters())

        param_groups = [
            {"params": [params_by_name[name] for name in group["names"]], **group["overrides"]}
            for group in self._optimizer_param_group_spec()
        ]

        return self.hparams.optimizer(params=param_groups)

    def get_losses(self, batch, is_training_step):
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
            is_training_step=is_training_step,
        )

        return diff_loss, dur_loss, prior_loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

        self._override_optimizer_params(checkpoint)

        self.add_speaker_if_needed(checkpoint)

    def _override_optimizer_params(self, checkpoint):
        """
        LR and weight_decay are saved in the checkpoint. If you want to change them in the yaml file, stop training and
        resume from a ckpt to use the new values, they need to be explicitly overridden after ckpt is loaded.

        Note that weight_decay is not the same for every parameter: embeddings, normalization parameters and biases are
        trained with weight_decay 0.0 (see _optimizer_param_group_spec) and have to stay at 0.0. So the value from the
        yaml is written only to the parameters that do use decay.
        """
        keywords = self.hparams.optimizer.keywords
        configured = {key: keywords[key] for key in ("lr", "weight_decay") if keywords.get(key) is not None}
        if not configured:
            return

        spec = self._optimizer_param_group_spec()

        for opt_state in checkpoint.get("optimizer_states", []):
            saved_groups = opt_state["param_groups"]
            saved_sizes = [len(group["params"]) for group in saved_groups]
            spec_sizes = [len(group["names"]) for group in spec]
            
            assert saved_sizes == spec_sizes, "Model changed, so this checkpoint cannot be resumed."

            # The saved groups were built from the spec, in the same order, so groups at the same position hold the
            # same parameters and the spec tells us which of them skip weight decay.
            for saved_group, spec_group in zip(saved_groups, spec):
                for key, config_val in configured.items():
                    group_val = spec_group["overrides"].get(key, config_val)
                    log.info("Overriding checkpoint %s %s with config value %s", key, saved_group.get(key), group_val)
                    saved_group[key] = group_val


    def add_speaker_if_needed(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        emb_keys = [k for k in ("speaker_embeddings_enc.weight", "speaker_embeddings_dur.weight") if
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

    METRIC_PRIOR = "prior"
    METRIC_DURATION = "duration"
    MAE_TRAIN_KEY = "mae/train_{}"
    MAE_VAL_KEY = "mae/val_{}"

    def _log_diagnostics(self, predictions, targets, mask, metric_name, is_training_step):
        """
        Logs diagnostic metrics for a loss term to help monitor training health and tune hyperparameters.

        During training, logs the Mean Absolute Error (MAE) and absolute error quantiles on the epoch
        before each validation epoch. The quantiles are useful for understanding the error distribution
        and for comparing against validation errors to assess generalization.
        Computing on the full epoch (rather than a subset of batches) gives a representative picture
        of the training error distribution. On all other training epochs, does nothing to avoid
        unnecessary computation.

        During validation, logs the MAE over the validation data. Combined with the training MAE
        logged on the preceding epoch, this allows computing the train-to-validation gap, which
        indicates how well the model generalizes.
        """
        if not self.hparams.log_diagnostics:
            return

        if is_training_step:
            is_epoch_before_validation = (self.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
            if not is_epoch_before_validation:
                return

            batch_size = predictions.shape[0]
            train_abs_error = torch.abs(predictions - targets)[mask.bool()]
            self.log(self.MAE_TRAIN_KEY.format(metric_name), train_abs_error.mean(), on_step=False, on_epoch=True, batch_size=batch_size)

            quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            quantiles_tensor = torch.tensor(quantiles, device=predictions.device)
            sorted_errors = torch.sort(train_abs_error).values
            n = sorted_errors.numel()
            indices = (quantiles_tensor * (n - 1)).long().clamp(0, n - 1)
            q = sorted_errors[indices]
            for i, p in enumerate(quantiles):
                self.log(f"abs_error_quantiles/{metric_name}_{p}", q[i], on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            batch_size = predictions.shape[0]
            valid_abs_error = torch.abs(predictions - targets)[mask.bool()]
            self.log(self.MAE_VAL_KEY.format(metric_name), valid_abs_error.mean(), on_step=False, on_epoch=True, batch_size=batch_size)

    def log_mae_gap(self, metrics, metric_name):
        if not self.hparams.log_diagnostics:
            return

        train_mae = metrics.get(self.MAE_TRAIN_KEY.format(metric_name))
        val_mae = metrics.get(self.MAE_VAL_KEY.format(metric_name))
        if train_mae is not None and val_mae is not None:
            gap_pct = (val_mae - train_mae) / train_mae * 100.0
            self.log(f"mae/gap_{metric_name}", gap_pct)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, 'create_batches'):
            old_len = len(sampler)
            sampler.create_batches()
            new_len = len(sampler)
            if old_len != new_len:
                log.error(f"Batch count changed from {old_len} to {new_len} at epoch {self.current_epoch}, this will cause Lightning to stop running validation.")

        # We log gap metrics at the start of the next training epoch after validation, because the callback_metrics are 
        # not yet available during on_validation_epoch_end or on_train_epoch_end.
        was_validation_epoch = self.current_epoch > 0 and self.current_epoch % self.trainer.check_val_every_n_epoch == 0
        if not was_validation_epoch:
            return

        # The gap metric says how much higher is the validation error compared to the training error, in percents.
        # A value of 25 means the validation loss is 25% worse than the training loss.
        metrics = self.trainer.callback_metrics
        self.log_mae_gap(metrics, self.METRIC_PRIOR)
        self.log_mae_gap(metrics, self.METRIC_DURATION)

    def training_step(self, batch: Any, batch_idx: int):
        diff_loss, dur_loss, prior_loss = self.get_losses(batch, is_training_step=True)
        bs = batch["x"].shape[0]
        # The 3 losses are independent, each influencing only its own part of the model, being detached
        # from the other parts. They are summed only because the optimizer needs a single number.
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
        diff_loss, dur_loss, prior_loss = self.get_losses(batch, is_training_step=False)
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

    def on_before_optimizer_step(self, optimizer):
        # Param and Grad norm computation is rather slow, so enable it only if you must see the charts in Tensorboard.
        if not self.hparams.log_diagnostics:
            return

        # self.encoder is torch.compile()'d, so its original submodules are accessed via _orig_mod.
        enc = self.encoder._orig_mod
        submodules = {
            "speaker_embeddings_enc": self.speaker_embeddings_enc,
            "speaker_embeddings_dur": self.speaker_embeddings_dur,
            "encoder":            self.encoder,
            "decoder":            self.decoder,
            "phoneme_embeddings": enc.emb,
            "enc_prenet":         enc.prenet,
            "enc_transformer":    enc.encoder,
            "enc_proj_m":         enc.proj_m,
            "enc_proj_w":         enc.proj_w,
        }
        for name, module in submodules.items():
            # Param norm helps me check if the weight decay value from Adam / AdamW is too large.
            # If param_norm stays flat or slightly increases: weight decay is just fine.
            # If param_norm is slowly sinking: weight decay is too big; it's slowly "erasing" the model.
            #
            # It also reveals which regularizer is doing the work when both weight decay and dropout are active.
            # If param_norm grows freely while overfitting stays under control, it means dropout is the dominant regularizer.
            param_norms = torch.stack([p.detach().norm() for p in module.parameters()])
            self.log(f"param_norm/{name}", torch.linalg.vector_norm(param_norms), on_step=False, on_epoch=True, logger=True, batch_size=1)

            params_with_grad = [p for p in module.parameters() if p.grad is not None]
            if params_with_grad:
                grad_norms = torch.stack([p.grad.norm() for p in params_with_grad])
                self.log(f"grad_norm/{name}", torch.linalg.vector_norm(grad_norms), on_step=False, on_epoch=True, logger=True, batch_size=1)

