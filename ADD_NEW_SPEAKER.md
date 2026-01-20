# Adding a New Speaker to a Trained Model

This guide explains how to add a new speaker to an already trained multi-speaker MatchaTTS model without affecting existing speakers.

## Overview

When you have a trained model with multiple speakers and want to add a new speaker, you can freeze the entire model (encoder, decoder, CFM) and train only the new speaker's embedding. This ensures:

- ✅ Existing speakers remain completely unchanged
- ✅ Fast training (only one embedding vector is updated)
- ✅ No risk of degrading existing speaker quality

## Implementation

### Step 1: Modify `baselightningmodule.py`

Edit the `configure_optimizers` method in `matcha/models/baselightningmodule.py`:

```python
def configure_optimizers(self) -> Any:
    # Check if training only a new speaker
    if hasattr(self, 'new_speaker_id'):
        params = [{'params': [self.spk_emb.weight[self.new_speaker_id]]}]
    else:
        params = self.parameters()
    
    optimizer = self.hparams.optimizer(params=params)
    if self.hparams.scheduler not in (None, {}):
        scheduler_args = {}
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
```

### Step 2: Prepare Your Data

1. Add the new speaker's audio files to your corpus
2. Update your corpus CSV file with the new speaker ID (e.g., if you had speakers 0-4, the new speaker is ID 5)
3. Update your corpus YAML to reflect `n_spks: 6` (or whatever the new total is)
4. Precompute mel spectrograms for the new speaker's data:

```bash
python -m matcha.utils.precompute_corpus -i configs/data/your-corpus.yaml
```

### Step 3: Load Model and Set New Speaker Mode

Before training, load your checkpoint with the increased speaker count and set the `new_speaker_id` attribute:

```python
# In your training script or notebook
model = MatchaTTS.load_from_checkpoint(
    "path/to/your/checkpoint.ckpt",
    n_spks=6  # Old count + 1
)

# Enable new speaker training mode
model.new_speaker_id = 5  # The ID of your new speaker
```

The `on_load_checkpoint` method in `baselightningmodule.py` automatically expands the speaker embedding layer to accommodate the new speaker.

### Step 4: Train

Run training as usual:

```bash
python -m matcha.train
```

Only the new speaker's embedding will be updated. All other model parameters remain frozen.

## Notes

- Use a smaller learning rate (e.g., 1e-4) since you're only training one embedding vector
- Training should be fast - you may only need 10-50 epochs depending on your data
- The new speaker embedding is initialized to zeros by default
- You can add multiple speakers at once by training with data from all new speakers simultaneously

## Verification

After training, test the new speaker:

```bash
python -m matcha.cli --text "Hello world" --checkpoint_path your-new-checkpoint.ckpt --spk 5
```

And verify existing speakers still sound the same:

```bash
python -m matcha.cli --text "Hello world" --checkpoint_path your-new-checkpoint.ckpt --spk 0
```
