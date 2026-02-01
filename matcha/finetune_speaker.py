"""
Fine-tune a trained MatchaTTS model to learn a new speaker or improve an existing one.

Usage:
    python -m matcha.finetune_speaker +target_speaker=0

Note: Configure data paths in your data yaml (e.g., speaker0.yaml)
      Set ckpt_path in train.yaml

Fine-tuning logic:
    1. Validate required config (ckpt_path and target_speaker)
    2. Set random seed
    3. Instantiate datamodule
    4. Instantiate fresh model from config
    5. Freeze encoder (includes duration predictor)
    6. Freeze decoder
    7. Register gradient hook to mask all speakers except target
    8. Instantiate callbacks
    9. Instantiate loggers
    10. Instantiate trainer
    11. Log hyperparameters
    12. Start training (trainer.fit loads checkpoint and trains)
    13. Log best checkpoint path
    14. Return metrics
"""

import torch
import types
import lightning as L
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matcha import utils

log = utils.get_pylogger(__name__)


def freeze_model_except_target_speaker(model, target_speaker):
    """Freeze all model parameters except the target speaker embedding."""
    # Set frozen modules to eval mode to disable dropout.
    # Otherwise, dropout noise would force the speaker embedding to overfit to random patterns
    # instead of learning actual speaker characteristics.
    log.info("Freezing encoder")
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    log.info("Freezing decoder")
    model.decoder.eval()
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    log.info(f"Training only speaker {target_speaker} embedding")
    
    def mask_speaker_gradients(grad):
        mask = torch.zeros_like(grad)
        mask[target_speaker] = 1.0
        return grad * mask
    
    model.spk_emb.weight.register_hook(mask_speaker_gradients)


@utils.task_wrapper
def finetune(cfg: DictConfig):
    # Required overrides
    if not cfg.get("ckpt_path"):
        raise ValueError("Must specify ckpt_path in train.yaml")
    if not cfg.get("target_speaker") and cfg.get("target_speaker") != 0:
        raise ValueError("Must specify +target_speaker=<speaker_id>")
    
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    freeze_model_except_target_speaker(model, cfg.target_speaker)
    
    # Save fingerprint of encoder/decoder weights for verification
    encoder_fingerprint = next(model.encoder.parameters()).clone()
    decoder_fingerprint = next(model.decoder.parameters()).clone()
    # I can delete the above lines after I tested few times

    # Override on_train_epoch_start to keep frozen modules in eval mode
    # trainer.fit() calls model.train() which resets all submodules to training mode
    original_on_train_epoch_start = model.on_train_epoch_start
    def on_train_epoch_start(self):
        if original_on_train_epoch_start:
            original_on_train_epoch_start()
        self.encoder.eval()
        self.decoder.eval()
    model.on_train_epoch_start = types.MethodType(on_train_epoch_start, model)
    
    log.info("Instantiating callbacks...")
    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))
    
    log.info("Instantiating loggers...")
    logger = utils.instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    
    if logger:
        log.info("Logging hyperparameters")
        utils.log_hyperparameters(object_dict)
    
    log.info(f"Fine-tuning speaker {cfg.target_speaker}")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)
    
    # Verify frozen weights didn't change
    encoder_unchanged = torch.equal(encoder_fingerprint, next(model.encoder.parameters()))
    decoder_unchanged = torch.equal(decoder_fingerprint, next(model.decoder.parameters()))
    log.info(f"Encoder weights unchanged: {encoder_unchanged}")
    log.info(f"Decoder weights unchanged: {decoder_unchanged}")
    if not encoder_unchanged or not decoder_unchanged:
        log.warning("WARNING: Frozen weights changed during training!")
    # I can delete the above lines after I tested a few times
    
    log.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer.callback_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    utils.extras(cfg)
    finetune(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
