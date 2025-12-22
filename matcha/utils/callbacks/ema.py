from torch_ema import ExponentialMovingAverage
from lightning.pytorch import Callback

class LightningEMA(Callback):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        # Initialize EMA with current model parameters
        self.ema = ExponentialMovingAverage(pl_module.parameters(), decay=self.decay)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update shadows after every optimizer step
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        # Swap in EMA weights for the validation run
        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(self, trainer, pl_module):
        # Restore original weights for training
        self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Crucial: Save the EMA state so you can resume!
        return {"ema_state_dict": self.ema.state_dict()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])