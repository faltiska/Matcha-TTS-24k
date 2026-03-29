#!/bin/bash

#  Mel Cepstral Distortion values per epoch
#  
#  Model/Epoch     V1/579          V2/139    V2/209    V2/259    V2/309   V2/424    V2/429    v4/869          v4/894    v4/934    v4/994 
#  speaker_000     5.66 dB         6.06 dB   5.84 dB   5.71 dB   5.24 dB  5.25 dB   5.24 dB   5.22 dB         5.17 dB   5.15 dB   5.30 dB
#  speaker_001     3.86 dB         4.57 dB   4.11 dB   3.99 dB   3.97 dB  4.04 dB   3.79 dB   3.80 dB         3.60 dB   3.70 dB   3.58 dB
#  speaker_002     4.24 dB         4.77 dB   4.47 dB   4.41 dB   4.33 dB  4.21 dB   4.17 dB   4.19 dB         4.15 dB   4.01 dB   3.94 dB
#  speaker_003     3.08 dB         4.06 dB   3.57 dB   3.55 dB   3.40 dB  3.22 dB   3.00 dB   2.66 dB         2.73 dB   2.72 dB   2.63 dB
#  speaker_004     5.99 dB         6.47 dB   6.50 dB   6.13 dB   6.08 dB  6.04 dB   5.96 dB   5.73 dB         5.77 dB   5.83 dB   5.70 dB
#  speaker_005     4.09 dB         4.53 dB   4.35 dB   4.14 dB   4.34 dB  4.22 dB   3.97 dB   3.89 dB         3.97 dB   3.91 dB   3.87 dB
#  speaker_006     3.84 dB         4.23 dB   4.00 dB   3.65 dB   3.85 dB  3.78 dB   3.68 dB   3.70 dB         3.61 dB   3.67 dB   3.68 dB
#  speaker_007     6.06 dB         6.77 dB   6.15 dB   5.75 dB   5.64 dB  5.53 dB   5.50 dB   5.72 dB         5.90 dB   5.79 dB   5.86 dB
#  speaker_008     5.50 dB         6.89 dB   6.23 dB   5.74 dB   5.54 dB  5.61 dB   5.57 dB   5.32 dB         5.60 dB   5.42 dB   5.31 dB
#  speaker_009     4.40 dB         4.96 dB   4.74 dB   4.61 dB   4.59 dB  4.41 dB   4.04 dB   4.47 dB         4.42 dB   4.28 dB   4.46 dB
#  --------------------------------------------------------------------------------------------------------------------------------------
#  Average         4.67 dB         5.33 dB   5.00 dB   4.77 dB   4.70 dB  4.63 dB   4.49 dB   4.47 dB         4.49 dB   4.45 dB   4.43 dB
#  
#  Model/Epoch     v6/89     v6/144    v6/189    v6/299    v6/414    V6/469    v6/519    v6/629    v6/729 
#  speaker_000     5.93 dB   5.51 dB   5.16 dB   5.39 dB   5.25 dB   5.09 dB   4.93 dB   4.93 dB   4.98 dB     
#  speaker_001     4.08 dB   3.99 dB   3.87 dB   3.74 dB   3.67 dB   3.72 dB   3.58 dB   3.55 dB   3.53 dB     
#  speaker_002     4.33 dB   4.33 dB   4.24 dB   4.16 dB   3.97 dB   3.86 dB   3.86 dB   3.85 dB   3.90 dB     
#  speaker_003     3.20 dB   3.19 dB   2.98 dB   2.90 dB   2.82 dB   2.75 dB   2.74 dB   2.62 dB   2.46 dB     
#  speaker_004     6.27 dB   6.16 dB   6.25 dB   5.86 dB   5.79 dB   5.65 dB   5.77 dB   5.89 dB   5.64 dB     
#  speaker_005     4.33 dB   4.38 dB   4.07 dB   4.04 dB   4.00 dB   3.97 dB   3.88 dB   3.80 dB   3.96 dB     
#  speaker_006     3.86 dB   4.03 dB   3.70 dB   3.87 dB   3.63 dB   3.51 dB   3.74 dB   3.67 dB   3.63 dB     
#  speaker_007     6.17 dB   5.95 dB   5.64 dB   5.57 dB   5.66 dB   5.52 dB   5.58 dB   5.43 dB   5.26 dB     
#  speaker_008     5.84 dB   5.67 dB   5.50 dB   5.31 dB   5.31 dB   5.27 dB   5.32 dB   5.20 dB   5.22 dB     
#  speaker_009     4.81 dB   4.81 dB   4.48 dB   4.28 dB   4.35 dB   4.33 dB   4.46 dB   4.32 dB   4.35 dB     
#  -------------------------------------------------------------------------------------------------------
#  Average         4.88 dB   4.80 dB   4.59 dB   4.51 dB   4.44 dB   4.37 dB   4.39 dB   4.33 dB   4.29 dB
#  
#  Model/Epoch     v8/134    v8/234    v8/339        v11/134   v11/144   v11/229
#  speaker_000     5.26 dB   5.26 dB   5.18 dB       5.27 dB   5.17 dB   5.21 dB     
#  speaker_001     3.84 dB   3.88 dB   3.57 dB       4.11 dB   3.96 dB   3.70 dB     
#  speaker_002     4.22 dB   3.97 dB   3.89 dB       4.38 dB   4.13 dB   4.17 dB     
#  speaker_003     3.07 dB   2.87 dB   2.76 dB       3.77 dB   3.56 dB   3.42 dB     
#  speaker_004     6.03 dB   6.04 dB   5.63 dB       5.83 dB   5.91 dB   5.77 dB     
#  speaker_005     4.12 dB   4.00 dB   3.92 dB       4.35 dB   4.05 dB   3.93 dB     
#  speaker_006     4.09 dB   3.78 dB   3.76 dB       3.83 dB   3.90 dB   3.68 dB     
#  speaker_007     5.83 dB   5.74 dB   5.67 dB       6.47 dB   5.87 dB   5.75 dB     
#  speaker_008     6.32 dB   5.58 dB   5.66 dB       5.75 dB   5.62 dB   5.61 dB     
#  speaker_009     4.59 dB   4.54 dB   4.23 dB       4.82 dB   4.64 dB   4.31 dB     
#  -----------------------------------------------------------------------------
#  Average         4.74 dB   4.56 dB   4.43 dB       4.86 dB   4.68 dB   4.56 dB

CHECKPOINTS=(
     "logs/train/v11/checkpoint_epoch=229-best-mcd.ckpt"
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
