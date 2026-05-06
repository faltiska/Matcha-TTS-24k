CHECKPOINT=logs/train/v19/runs/2026-05-06_12-56-17/checkpoints/checkpoint_epoch=004.ckpt

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "The rain continued its relentless assault against the windowpane, each drop a tiny, percussive reminder of the time slipping through his fingers like sand in an hourglass that someone had forgotten to turn, and as he sat there, the steam from his tea curling into the stagnant air of the study, he found himself wandering back through the labyrinth of his own memory, passing by the shuttered doors of old regrets and the brightly lit hallways of summers long since faded into the grey hues of nostalgia." \
--spk "0,1,2,3,4,5,6,10,11,12" \
--debug 

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "Ploaia continua să cadă neîntrerupt peste orașul, fiecare picătură amintind de timpul care trece." \
--spk "7" \
--debug 

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "La pluie continuait son assaut implacable contre la vitre, chaque goutte un rappel du temps qui s'écoule." \
--spk "8,9" \
--debug 

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "La pioggia continuava il suo assalto incessante contro il vetro, ogni goccia un promemoria del tempo che scorre." \
--spk "13,14" \
--debug 
