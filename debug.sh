CHECKPOINT=logs/train/v20/checkpoint_epoch=054.ckpt

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "The rain continued its relentless assault against the windowpane, each drop a tiny, percussive reminder of the time slipping through his fingers. Plus, there was always Bubba’s drinking-and-driving problem, which Tommy was having to handle far too frequently lately." \
--spk "0,1,2,3,4,5,6,10,11,12" \
--debug 

python -m matcha.cli \
--checkpoint_path $CHECKPOINT \
--text "Ploaia continua să cadă neîntrerupt peste oraș, fiecare picătură amintind de timpul care trece." \
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
