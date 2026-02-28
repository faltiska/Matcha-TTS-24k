#!/bin/bash

#  Mel Cepstral Distortion values per epoch
#  
#  Model/Epoch     V1/249    V1/579          V2/139    V2/209    V2/259    V2/309   V2/424    V2/429    V2/564    V2/629 
#  speaker_000     5.97 dB   5.66 dB         6.06 dB   5.84 dB   5.71 dB   5.24 dB  5.25 dB   5.24 dB   5.38 dB   5.20 dB
#  speaker_001     4.30 dB   3.86 dB         4.57 dB   4.11 dB   3.99 dB   3.97 dB  4.04 dB   3.79 dB   3.89 dB   3.91 dB
#  speaker_002     4.55 dB   4.24 dB         4.77 dB   4.47 dB   4.41 dB   4.33 dB  4.21 dB   4.17 dB   4.07 dB   4.20 dB
#  speaker_003     3.68 dB   3.08 dB         4.06 dB   3.57 dB   3.55 dB   3.40 dB  3.22 dB   3.00 dB   2.91 dB   3.00 dB
#  speaker_004     6.57 dB   5.99 dB         6.47 dB   6.50 dB   6.13 dB   6.08 dB  6.04 dB   5.96 dB   5.88 dB   5.76 dB
#  speaker_005     4.51 dB   4.09 dB         4.53 dB   4.35 dB   4.14 dB   4.34 dB  4.22 dB   3.97 dB   4.18 dB   4.00 dB
#  speaker_006     4.17 dB   3.84 dB         4.23 dB   4.00 dB   3.65 dB   3.85 dB  3.78 dB   3.68 dB   3.72 dB   3.76 dB
#  speaker_007     6.58 dB   6.06 dB         6.77 dB   6.15 dB   5.75 dB   5.64 dB  5.53 dB   5.50 dB   5.61 dB   5.61 dB
#  speaker_008     6.12 dB   5.50 dB         6.89 dB   6.23 dB   5.74 dB   5.54 dB  5.61 dB   5.57 dB   5.55 dB   5.42 dB
#  speaker_009     4.64 dB   4.40 dB         4.96 dB   4.74 dB   4.61 dB   4.59 dB  4.41 dB   4.04 dB   4.32 dB   4.30 dB
#  ----------------------------------------------------------------------------------------------------------------------
#  Average         5.11 dB   4.67 dB         5.33 dB   5.00 dB   4.77 dB   4.70 dB  4.63 dB   4.49 dB   4.55 dB   4.52 dB
 
CHECKPOINTS=(
     "logs/train/v4/runs/2026-02-27_21-45-05/checkpoints/checkpoint_epoch=394.ckpt"
)
  
for CHECKPOINT_PATH in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename "$CHECKPOINT_PATH" .ckpt)
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
        exit 1
    fi
    
    echo "Processing $CKPT_NAME..."
    
    rm -f mcd_validation/utterance_*.wav
    
    _TMP=$(mktemp)
    run_cli() { python -m matcha.cli "$@" --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --output_folder mcd_validation > "$_TMP" 2>&1 || { cat "$_TMP"; rm -f "$_TMP"; exit 1; }; }

    run_cli --text "There is a strong chance it will happen once more." --spk "0,1,2,3,4,5,6"
    run_cli --text "Elles doivent simplement être précisées par décret." --spk "8,9"
    run_cli --text "Geologul analizează geoda găsită în groapa adâncă din grădină." --spk "7"
    rm -f "$_TMP"
    
    python -m matcha.utils.compute_mcd mcd_validation
    echo "Completed $CKPT_NAME"
    echo "---"
done
