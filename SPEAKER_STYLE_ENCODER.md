# Adding New Speakers with the StyleEncoder

The StyleEncoder lets you add a new speaker to an existing MatchaTTS checkpoint
without retraining the full model.

It works in two stages:

1. **Train the StyleEncoder once** against a good MatchaTTS checkpoint.
   The encoder learns to predict the three speaker embedding vectors (encoder, duration predictor, decoder)
   from mel spectrograms.

2. **Add a new speaker** by running their audio through the trained encoder.
   The predicted embeddings are appended to the checkpoint's embedding tables and saved as a new checkpoint.
   The new speaker gets the next available speaker ID.

---

## Stage 1 — Train the StyleEncoder

Do this once. You need a good MatchaTTS checkpoint and the precomputed mel cache for your corpus.

All configuration lives in the yaml files — no CLI arguments are required for a standard run:

```bash
python -m matcha.train_style_encoder
```

The MatchaTTS checkpoint, corpus path, and mel directory are set in:
- `configs/train_style_encoder.yaml` — `matcha_ckpt`, `run_name`
- `configs/data/speaker-style-encoder.yaml` — `train_filelist_path`, `mel_dir`, `refs_per_step`
- `configs/model/style_encoder/default.yaml` — `n_channels`
- `configs/model/optimizer/style_encoder_adamw.yaml` — `lr`, `weight_decay`

Override any value on the command line if needed:

```bash
python -m matcha.train_style_encoder \
    matcha_ckpt=logs/train/v6/checkpoint.ckpt \
    model.optimizer.lr=5e-5
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/train_style_encoder
```

Key config values in `configs/model/style_encoder/default.yaml`:
- `n_channels` — internal feature size of the Conv stack and GRU (default: 128)

Key config values in `configs/data/speaker-style-encoder.yaml`:
- `refs_per_step` — number of reference mels aggregated per training step (default: 32)

Optimizer and learning rate are configured in `configs/model/optimizer/style_encoder_adamw.yaml`.

Checkpoints are saved to `logs/train_style_encoder/runs/<timestamp>/checkpoints/`
by the standard `ModelCheckpoint` callback.

---

## Stage 2 — Add a New Speaker

Prepare mel spectrograms for the new speaker using the precompute script:

```bash
python -m matcha.utils.precompute_corpus -i configs/data/my-new-speaker.yaml
```

Then run:

```bash
python -m matcha.utils.add_speaker \
    --matcha_ckpt logs/train/v4/checkpoint_epoch=1189.ckpt \
    --style_encoder_ckpt logs/train_style_encoder/v1/runs/2026-03-07_15-10-13/checkpoints/checkpoint_epoch=044.ckpt \
    --mel_dir data/corpus-small-24k/mels/10 \
    --output_ckpt new_speaker.ckpt
```

The script will:
- Encode all `.npy` mel files found recursively under `--mel_dir`
- Predict the three speaker embedding vectors
- Append them as a new row to the checkpoint's embedding tables
- Increment `n_spks` in the checkpoint's hyperparameters
- Save the updated checkpoint to `--output_ckpt`

The new speaker's ID will be printed during the run. Use it with `--spk` at inference time.

More reference mels generally produce better embeddings. There is no strict minimum,
but covering a range of phonetic content and prosodic variation will give better results.

---

Test the new speaker with
```
python -m matcha.cli \
--text "There is a strong chance it will happen once more." \
--spk 10 --vocoder vocos \
--checkpoint_path new_speaker.ckpt
```

---

## Notes

- The StyleEncoder is tied to the MatchaTTS checkpoint it was trained against.
  If you train a new MatchaTTS model, retrain the encoder too.
- The `add_speaker` script does not modify the source checkpoint. It always writes a new file.
- The new speaker's embeddings are predicted from audio alone, without fine-tuning.
  For best quality, consider following up with a short fine-tuning run using
  `python -m matcha.finetune_speaker` targeting the new speaker ID.
