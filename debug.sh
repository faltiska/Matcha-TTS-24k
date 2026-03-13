CHECKPOINT_PATH=logs/train/v6/checkpoint_epoch=949.ckpt

python -m matcha.cli --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --text "OK, I hear you. Can we go now?" --spk "6" --debug