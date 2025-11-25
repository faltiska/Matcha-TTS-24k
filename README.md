1. Create an UV environment

```
uv venv --python 3.10
.venv\Scripts\activate
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
python setup.py build_ext --inplace
```

2. Inference
Set this env var on each terminal where you want to run inference:
```
set "PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll"
```
If in IntelliJ Idea, you can set it in the terminal settings, and Idea will run it for you.

Run inference with:
```
python -m matcha.cli --text "Are you listening?"
```

I have added a new Vocoder, Vocos, used with a model trained on 24KHz audio.  

3. Training
Delete the mels and f0 folders from the corpus, if they exist.
Compute statistics for the corpus and update the corpus yaml with te stats:
```
 matcha-data-stats -i configs/data/corpus-small-24k.yaml -f
```
It will output something like
{'mel_mean': -1.7744582891464233, 'mel_std': 4.116815090179443}
{'f0_mean': 248.1585693359375, 'f0_std': 297.7625427246094}
Take the values and put them in the yaml file.

Precompute mel spectrogram cache to speed up training. 
This avoids recomputing mels every epoch and reduces data-loading overhead.
```
python -m matcha.utils.precompute_corpus -i configs/data/corpus-small-24k.yaml
```

You can test the precomputed mel files using the vocoder wrapper scripts.
For a 22KHz corpus, and the HiFiGAN vocoder:
```
python -m matcha.hifigan.models --wav input.wav --data-config  configs/data/corpus-small-22k.yaml --vocoder-id hifigan_T2_v1
python -m matcha.hifigan.models --mel input.mel --data-config  configs/data/corpus-small-22k.yaml --vocoder-id hifigan_T2_v1
```
which will output a file called vocoder-test.wav in the project root.

For a 24KHz corpus, and Vocos:
```
python -m matcha.vocos24k.wrapper --wav input.wav --data-config  configs/data/corpus-small-24k.yaml
python -m matcha.vocos24k.wrapper --mel input.mel --data-config  configs/data/corpus-small-24k.yaml
```
which will output a file called vocoder-test24k.wav

You can compare the reconstituted files to the original input wav, to asses the quality of the vocoder.
I find hifigan_T2_v1 better than hifigan_univ_v1, and Vocos at 24KHz better than hifigan_T2_v1.


Prepare your corpus, update configs/train.yaml, then run:
```
python -m matcha.train
```   


See original [readme](ORIGINAL-README.md) too.