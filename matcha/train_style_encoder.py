import os
from pathlib import Path

cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

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


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

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
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_style_encoder.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    return metric_value


if __name__ == "__main__":
    main()
