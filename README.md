## Create an UV environment

```
uv venv --python 3.10
.venv\Scripts\activate
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
python setup.py build_ext --inplace --force
uv pip install git+https://github.com/supertone-inc/super-monotonic-align.git
```

## Inference
Run inference with:
```
python -m matcha.cli --text "You're leaving?"
python -m matcha.cli --text "You're leaving?" --vocoder vocos --checkpoint_path <your-chekpoint.ckpt> --spk 0,1,2,3
```

## Training
Delete the mels folders from the corpus, if they exist.
Compute statistics for the corpus and update the corpus yaml with te stats:
```
matcha-data-stats -i corpus-small-24k.yaml
```
It will output something like
{'mel_mean': -1.7744582891464233, 'mel_std': 4.116815090179443}
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

Monitor training with: 
```
tensorboard --logdir logs/train/corpus-small-24k/runs/2025-11-26_09-03-10/tensorboard/version_0
```

Profile your trainer with:
```
python -m matcha.train +trainer.profiler="simple"
```
This will generate a profile report at the end of training, so maybe set it to run for just a small number of epochs.

## Improvements

Compared to the original MatchaTTS, I did the following:
- increased the decoder model capacity
- increased the TextEncoder model capacity
- switched to an AdamW optimizer
- added Vocos using a model trained on 24KHz audio 
- increased the TextEncoder model size
- implemented a corpus mel precomputation script
- included a matmul precision auto-config
- added 2 more ODE solvers 
- switched to the Super-MAS monotonic_align implementation
- found a series of performance improvements 
- made changes to get the model to compile

# Learning gradients 

The gradient norm charts tell you about training stability and convergence:
- What gradient norms mean:
- Total norm: Overall magnitude of all gradients combined 
- Too high (>10): Gradients exploding, model unstable 
- Too low (<0.001): Gradients vanishing, model not learning 
- Healthy: 0.1-5 range, gradually decreasing over time

Per-layer norms: Which parts of your model are learning
- Encoder vs Decoder vs Flow matching components 
- If one is much larger, that component dominates learning
- If one is near zero, that component isn't learning

What to look for:
- Spikes: Sudden jumps indicate instability - may need lower learning rate or gradient clipping 
- Flat lines: Model stopped learning - learning rate too low or saturated
- Steady decrease: Good! Model is converging smoothly
- Oscillations: Normal early on, but persistent oscillations suggest learning rate too high

# PyTorch stuff

When compiling models, pytorch stores some information to be reused at later runs.
Triton kernel cache:
- ~/.triton/cache/

TorchInductor cache: 
- ~/.cache/torch/
- /tmp/torchinductor_$USER/

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


