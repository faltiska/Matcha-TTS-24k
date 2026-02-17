## Environment preparation

```
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install torch torchaudio torchvision torchcodec --index-url https://download.pytorch.org/whl/cu130 --upgrade
  or use the 2.9.1 version, it was slightly faster than 0.10.0
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchcodec==0.9.1 --index-url https://download.pytorch.org/whl/cu130 --upgrade
uv pip install -r requirements.txt --upgrade
python setup.py build_ext --inplace --force
uv pip install git+https://github.com/supertone-inc/super-monotonic-align.git --upgrade
uv pip install -e .
```

## Inference
Run inference with:
```
python -m matcha.cli --text "You're leaving?" --vocoder vocos --checkpoint_path <your-chekpoint.ckpt> --spk 0 --language en-us
```

## Training

### Check your corpus
Filter out files that are longer than N seconds
```
python -m matcha.utils.filter_by_wav_duration data/corpus-small-24k/train.csv 12
python -m matcha.utils.filter_by_wav_duration data/corpus-small-24k/validate.csv 12
```

Check if the corpus uses any unknown IPA symbol. 
eSpeak could generate a symbol we do not have in our symbols.py map.
```
python -m matcha.utils.validate_corpus_ipa data/corpus-small-24k/train.csv
python -m matcha.utils.validate_corpus_ipa data/corpus-small-24k/validate.csv
```

Measure trailing silence per speaker to check for inconsistencies.
Different speakers may have different amounts of trailing silence, which can cause the model to learn silence duration as a spurious feature for speaker identification 
instead of actual voice characteristics.
```
python -m matcha.utils.measure_silence -i configs/data/corpus-small-24k.yaml
```
If you see significant variation in trailing silence between speakers (e.g., some speakers with ~300ms and others with ~900ms), you should normalize it.
Backup your wav files first, then add silence to reach a consistent target duration:
```
python -m matcha.utils.add_silence -i configs/data/corpus-small-24k.yaml --target_silence 0.8
```
The script measures current silence at -60dB threshold and adds only what's needed to reach the target.
After running, verify the normalization by measuring again - all speakers should now have the same trailing silence duration.

### Calculate corpus statistics
Delete the mels folders from the corpus, if they exist.
Compute statistics for the corpus and update the corpus yaml with te stats:
```
matcha-data-stats -i corpus-small-24k.yaml
```
It will output something like
{'mel_mean': -1.7744582891464233, 'mel_std': 4.116815090179443}
Take the values and put them in the yaml file.

### Prepare spectrograms
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
python -m matcha.vocos24k.vocos_wrapper --wav input.wav --data-config  configs/data/corpus-small-24k.yaml
python -m matcha.vocos24k.vocos_wrapper --mel input.mel --data-config  configs/data/corpus-small-24k.yaml
```
which will output a file called vocoder-test24k.wav

You can compare the reconstituted files to the original input wav, to asses the quality of the vocoder.
I find hifigan_T2_v1 better than hifigan_univ_v1, and Vocos at 24KHz better than hifigan_T2_v1.

### Train
```
python -m matcha.train
```   

Monitor training with: 
```
tensorboard --logdir logs/train/corpus-small-24k/runs/2025-11-26_09-03-10/tensorboard/version_0
```

Profile your trainer with:
```
python -m matcha.train +trainer.profiler="simple"
```
This will generate a profile report at the end of training, so maybe set it to run for just a small number of epochs.

See how much audio you have in the corpus:
```
python -m matcha.utils.compute_corpus_duration data/corpus-small-24k/train.csv data/corpus-small-24k/validate.csv
```
### Fine tune a speaker or add a new one:
```
python -m matcha.finetune_speaker \
    +new_speaker_train_csv=data/new_speaker/train.csv \
    +new_speaker_val_csv=data/new_speaker/val.csv \
    +new_speaker_id=10
```

## Improvements

Compared to the original MatchaTTS, I did the following:
- Switched to Vocos using a model trained on 24KHz audio
  All other vocoders were trained on 22KHz audio files, and should have had an f_max of 11KHz
  but f_max was set to 8KHz for all of them.  
- Increased the decoder model capacity
  I hope it will make room for the extra frequencies that came with using Vocos 24K.
- Increased the spk_emb_dim in the encoder model
  I hope it will capture the differences between speakers better
- Switched to an AdamW optimizer
- Implemented a mel precomputation script
  Originally, the mels were computed during training, on the fly.
  This speeds up training a bit
- Switched to the torch built in ODE solver 
  No need to maintain our own version, since the torch one supports all algorithms. 
- Made some performance improvements to the CPU based monotonic_align implementation
- Switched to the Super-MAS monotonic_align implementation (you can find it in GitHub)
  It is fast, but not really faster than the CPU version
- Found a series of other performance improvements 
- Made changes to get the model to compile, this is the biggest performance improvement.

# PyTorch stuff

When compiling models, pytorch stores some information to be reused at later runs.
You could delete the folders to clear the cache:
```
rm -rf ~/.triton/cache/
rm -rf ~/.cache/torch/
rm -rf /tmp/torchinductor_$USER/
rm -rf ~/.nv/ComputeCache/
rm -rf ~/.cache/torch_extensions/ 
```

# nVidia drivers

Update the drivers with:
```
sudo apt install cuda-drivers --update
sudo apt install libcudnn9-cuda-13 --update
```
or update all linux packages:
```
sudo apt update && sudo apt full-upgrade
```

# Misc

See original [readme](ORIGINAL-README.md) too.