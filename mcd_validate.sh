#!/bin/bash

CHECKPOINT_PATH="logs/train/corpus-small-24k/runs/2026-02-16_11-22-14/checkpoints/saved/checkpoint_epoch=529.ckpt"

python -m matcha.cli --text "There is a strong chance it will happen once more." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "0,1,2,6" --language en-us --output_folder mcd_validation
python -m matcha.cli --text "There is a strong chance it will happen once more." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "3,4,5" --language en-gb --output_folder mcd_validation

python -m matcha.cli --text "Elles doivent simplement être précisées par décret." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "8,9" --language fr-fr --output_folder mcd_validation
python -m matcha.cli --text "Geologul analizează geoda găsită în groapa adâncă din grădină." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "7" --language ro --output_folder mcd_validation

python -m matcha.utils.compute_mcd ./mcd_validation