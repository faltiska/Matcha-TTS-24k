#!/bin/bash


CHECKPOINTS=(
    "logs/train/corpus-small-24k/runs/2026-02-13_17-54-59/checkpoints/saved/checkpoint_epoch=059.ckpt"
    "logs/train/corpus-small-24k/runs/2026-02-13_17-54-59/checkpoints/saved/checkpoint_epoch=179.ckpt"
    "logs/train/corpus-small-24k/runs/2026-02-15_10-21-46/checkpoints/saved/checkpoint_epoch=299.ckpt"
    "logs/train/corpus-small-24k/runs/2026-02-15_20-10-46/checkpoints/saved/checkpoint_epoch=429.ckpt"
    "logs/train/corpus-small-24k/runs/2026-02-16_11-22-14/checkpoints/saved/checkpoint_epoch=579.ckpt"
    "logs/train/corpus-small-24k/runs/2026-02-16_11-22-14/checkpoints/saved/checkpoint_epoch=639.ckpt"
)

for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
        exit 1
    fi
    
    echo "Processing $CKPT_NAME..."
    
    rm -f mcd_validation/utterance_*.wav
    
    python -m matcha.cli --text "There is a strong chance it will happen once more." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "0,1,2,6" --language en-us --output_folder mcd_validation > /dev/null 2>&1 || { echo "ERROR: Generation failed"; exit 1; }
    python -m matcha.cli --text "There is a strong chance it will happen once more." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "3,4,5" --language en-gb --output_folder mcd_validation > /dev/null 2>&1 || { echo "ERROR: Generation failed"; exit 1; }
    
    python -m matcha.cli --text "Elles doivent simplement être précisées par décret." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "8,9" --language fr-fr --output_folder mcd_validation > /dev/null 2>&1 || { echo "ERROR: Generation failed"; exit 1; }
    python -m matcha.cli --text "Geologul analizează geoda găsită în groapa adâncă din grădină." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "7" --language ro --output_folder mcd_validation > /dev/null 2>&1 || { echo "ERROR: Generation failed"; exit 1; }
    
    python -m matcha.utils.compute_mcd mcd_validation
    echo "Completed $CKPT_NAME"
    echo "---"
done
