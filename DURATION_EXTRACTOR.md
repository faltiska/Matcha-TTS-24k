# ASR-Based Duration Extraction for MatchaTTS

## Requirements

### Objective
Create script to extract phoneme durations using StyleTTS2's ASR pipeline.
Use the corpus in data/corpus-small-24k. You can find a clear description of the CSV file in the precompute_corpus.py script.
The StyleTTS2 code is temporarily copied to the folder StyleTTS2-main in project root.
The ASR code is permanently copied to matcha/utils/ASR. That will be committed to Git.
It is acceptable to modify it if needed. The new scripts must be in the same folder.

The extracted durations will be used with MatchaTTS's `use_precomputed_durations=True` training mode.
They must be compatible with the Matcha model (sample rate, frame length, whatever else applies).

### Constraints & Existing Infrastructure

1. **Mel spectrograms** - Already exist at `data/corpus-small-24k/mels/{rel_path}.npy`
   The Matcha TTS mel spectrograms are no compatible with the ASR model.
   The new script should create its own set of mels for duration extraction, in its format.  
   It should not overwrite the mels that will be used in Matcha training.

2. **Matcha's text-to-IPA conversion** - Use `matcha.text.to_phonemes(text, language)`
   The IPA conversion table in Matcha TTS uses more symbols than were used in training the ASR model.
   I should train an ASR model on my corpus.

3. **GPU-based MAS like Matcha does** - Use `super_monotonic_align.maximum_path(scores, mask, dtype)`
See matcha_tts.py for how the super-mas is used in MAtcha. 

#### Corpus Format
- CSV file: `rel_path|spk_id|language|text`
- Example: `0/0001|0|en-us|Hello world`
- Mel files: `corpus_dir/mels/{rel_path}.npy`
- Output durations: `corpus_dir/durations/{rel_path}.pt`

### Technical Specifications

#### ASR Model Integration
- **Model**: StyleTTS2's ASRCNN from `matcha/utils/ASR/models.py`, if possible
- **Checkpoint**: `matcha/utils/ASR/epoch_00080.pth`
- **Config**: `matcha/utils/ASR/config.yml`

If the model is not compatible, and we cannot reconciliate the differences, then we could try to train our own.

#### Processing Pipeline
1. Load precomputed mel: `torch.from_numpy(np.load(mel_path))`
2. Convert text to IPA: `matcha.text.to_phonemes(text, language)`
3. Convert IPA to token IDs using `matcha.text.symbols`
4. Call ASR model: `asr_model(mel, mask, tokens)`
5. Process attention the same way StyleTTS2 does
6. Apply MAS: `hard_alignment = maximum_path(s2s_attn, mask, dtype)`
7. Extract durations: `asr_durations = hard_alignment.sum(dim=-1)`
9. Save: `torch.save(target_durations, duration_path)`

### Output Format

Must save durations in the format that Matcha expects to be able to load them for training. 

#### Matcha's Expected Format
See `matcha_tts.py` line 207:
```python
if self.use_precomputed_durations:
    attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
```

The `durations` parameter is passed to `generate_path()` which expects:
- **Shape**: `(batch_size, text_length)` or `(text_length,)` for single sample
- **Type**: `torch.Tensor` with integer values
- **Values**: Number of mel frames each phoneme should last
- **Units**: Matcha's frame units (hop_length=256)

### Script Requirements

#### Command Line Interface
```bash
python -m matcha.utils.extract_durations --corpus <path> [--device cuda]
```
- Only `--corpus` is required
- `--device` defaults to `cuda`
All other values can be hardcoded: 
- ASR checkpoint: `matcha/utils/ASR/epoch_00080.pth`
- ASR config: `matcha/utils/ASR/config.yml`
- Target hop length: 256
- Output dir: Derived from corpus path as `corpus_dir/durations`

#### Error Handling
Report error and stop on: 
- missing mel files
- unknown IPA symbols
Print success/failure counts at end


## Alternative solution

I could try Wav2Vec2 instead of ASR. 