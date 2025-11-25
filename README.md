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
I have also added code to test the vocoders, using these commands:
```
python -m matcha.hifigan.models --wav input.wav --vocoder-id hifigan_T2_v1
```
which will output a file called vocoder-test.wav in the project root.
```
python -m matcha.vocos24k.wrapper --wav input.wav
```
which will output a file called vocoder-test24k.wav

You can compare the reconstituted files to the original input wav, to asses the quality of the vocoder. 
I find hifigan_T2_v1 better than hifigan_univ_v1, and Vocos at 24KHz better than hifigan_T2_v1.   

3. Training

Precompute mel spectrogram cache to speed up training. 
This avoids recomputing mels every epoch and reduces data-loading overhead.
```
python -m matcha.utils.precompute_corpus -i configs/data/your-corpus.yaml
```

Prepare your corpus. Update configs/train.yaml. Run:
```
python -m matcha.train
```   


See original [readme](ORIGINAL-README.md) too.