#!/bin/bash

CHECKPOINT_PATH="logs/train/corpus-small-24k/runs/2026-01-28_10-51-42/checkpoints/saved/checkpoint_epoch=1309.ckpt"

python -m matcha.cli --text "It was a dark and stormy night; the rain fell in torrents—except at occasional intervals, when it was checked by a violent gust of wind which swept up the streets (for it is in London that our scene lies), rattling along the housetops, and fiercely agitating the scanty flame of the lamps that struggled against the darkness." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "0,1,2,6" --language en-us

python -m matcha.cli --text "It was a dark and stormy night; the rain fell in torrents—except at occasional intervals, when it was checked by a violent gust of wind which swept up the streets (for it is in London that our scene lies), rattling along the housetops, and fiercely agitating the scanty flame of the lamps that struggled against the darkness." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "3,4,5" --language en-gb

python -m matcha.cli --text "C'était une nuit sombre et orageuse; la pluie tombait à torrents—sauf à intervalles occasionnels, quand elle était arrêtée par une violente rafale de vent qui balayait les rues (car c'est à Londres que se déroule notre scène), claquant le long des toits, et agitant férocement la maigre flamme des lampes qui luttaient contre l'obscurité." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "8,9" --language fr-fr

python -m matcha.cli --text "Era o noapte întunecată și furtunoasă; ploaia cădea în torente—cu excepția momentelor ocazionale când era oprită de o rafală violentă de vânt care mătura străzile (căci în Londra se petrece scena noastră), zăngănind peste acoperișuri și agitând cu furie flacăra slabă a lămpilor care se luptau cu întunericul." --vocoder vocos --checkpoint_path "$CHECKPOINT_PATH" --spk "7" --language ro
