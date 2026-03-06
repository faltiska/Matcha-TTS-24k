"""
Fine-tune speaker embeddings only, leaving the rest of the model frozen.

Fine-tune an existing speaker:
    python -m matcha.finetune_speaker +target_speaker=3

Add a new speaker (must also set n_spks in data config):
    python -m matcha.finetune_speaker +target_speaker=10 data.n_spks=11
"""
import os
from pathlib import Path

cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

import types
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import logging

from matcha import utils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


SPK_EMB_NAMES = ("encoder_speaker_embeddings.weight", "duration_speaker_embeddings.weight", "decoder_speaker_embeddings.weight")


def freeze_all_except_target_speaker(model: LightningModule, target_speaker: int):
    for name, param in model.named_parameters():
        param.requires_grad = name in SPK_EMB_NAMES

    def _mask_grad(grad):
        masked = torch.zeros_like(grad)
        masked[target_speaker] = grad[target_speaker]
        return masked

    for emb in (model.encoder_speaker_embeddings, model.duration_speaker_embeddings, model.decoder_speaker_embeddings):
        emb.weight.register_hook(_mask_grad)
    log.info(f"Unfrozen spk_emb_encoder/duration/decoder row {target_speaker} only. All other parameters frozen.")



def filter_dataset_to_speaker(datamodule: LightningDataModule, target_speaker: int):
    for split, dataset in [("train", datamodule.trainset), ("val", datamodule.validset)]:
        before = len(dataset.filepaths_and_text)
        dataset.filepaths_and_text = [
            row for row in dataset.filepaths_and_text if int(row[1]) == target_speaker
        ]
        after = len(dataset.filepaths_and_text)
        log.info(f"{split}: filtered {before} -> {after} samples for speaker {target_speaker}")


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    target_speaker: int = cfg.target_speaker

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.sample_rate = cfg.data.sample_rate
    model.hop_length = cfg.data.hop_length

    ckpt_path = cfg.get("ckpt_path")

    original_on_load_checkpoint = model.on_load_checkpoint

    def patched_on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        original_on_load_checkpoint(checkpoint)
        all_param_names = [name for name, _ in self.named_parameters()]
        spk_emb_indices = [all_param_names.index(n) for n in all_param_names if n in SPK_EMB_NAMES]
        for opt_state in checkpoint.get("optimizer_states", []):
            old_state = opt_state.get("state", {})
            opt_state["state"] = {
                new_idx: old_state.get(old_idx, old_state.get(str(old_idx)))
                for new_idx, old_idx in enumerate(spk_emb_indices)
                if old_idx in old_state or str(old_idx) in old_state
            }
            for param_group in opt_state.get("param_groups", []):
                param_group["params"] = list(range(len(spk_emb_indices)))

    model.on_load_checkpoint = types.MethodType(patched_on_load_checkpoint, model)

    def patched_configure_optimizers(self):
        spk_emb_params = [p for n, p in self.named_parameters() if n in SPK_EMB_NAMES]
        return self.hparams.optimizer(params=spk_emb_params)

    model.configure_optimizers = types.MethodType(patched_configure_optimizers, model)

    freeze_all_except_target_speaker(model, target_speaker)

    datamodule.setup()
    filter_dataset_to_speaker(datamodule, target_speaker)
    datamodule.setup = lambda stage=None: None  # prevent Lightning from re-running setup

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        utils.log_hyperparameters(object_dict)

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    return utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
