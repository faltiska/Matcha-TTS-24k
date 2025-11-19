Implemented a supervised F0 (pitch) conditioning with a pitch loss, wired into training and inference paths.

1. Text encoder: added a pitch head

    - File: matcha/models/components/text_encoder.py
    - Added PitchPredictor mirroring DurationPredictor.
    - TextEncoder.forward now returns (mu, logw, x_mask, p_log) where p_log is predicted log-F0 at text resolution.

2. Data pipeline: Added F0 extraction

    - File: matcha/data/text_mel_datamodule.py
    - Added get_f0 using torchaudio.functional.detect_pitch_frequency (YIN). Produces per-frame F0 aligned to mel frames using hop_length.
    - Batch contains keys: "f0" (B, 1, T_mel) and "f0_mask" (B, 1, T_mel), where mask=1 for voiced frames (F0>0).

3. Model: added conditioning + loss

    - File: matcha/models/matcha_tts.py
    - New hparams: use_pitch (default true in config), lambda_pitch (default 0.2).
    - If use_pitch:
        - Added f0_proj: Conv1d(1 → n_feats) to inject per-frame F0 embedding into mu_y.
        - Training forward accepts f0, f0_mask; adds f0_proj(f0) to mu_y before decoder.
        - Pitch loss: MSE between aligned p_log (text->mel via attn) and log(f0+1e-8) masked by f0_mask and y_mask. 
          Added to total loss scaled by lambda_pitch.
    - Inference uses predicted p_log aligned with attn to produce f0_est = exp(p_log_y) and conditions mu_y with f0_proj(f0_est) for better prosody.

4. Lightning base: logging integration

    - File: matcha/models/baselightningmodule.py
    - get_losses now reads batch["f0"], batch["f0_mask"] when present.
    - Logs sub_loss/train_pitch_loss and sub_loss/val_pitch_loss if model.use_pitch is true.
    - Total loss includes pitch_loss automatically via sum(loss_dict.values()).

5. Config: enable pitch conditioning

    - File: configs/model/matcha.yaml
    - Added:
        - use_pitch: true
        - lambda_pitch: 0.2

6. How it fits with the original code:

    - Uses the existing conditioning interfaces in the decoder stack and alignment map to place F0 at the mel-frame level.
    - Adds a light head to the encoder (no change to UNet shapes).
    - Keeps the 3 original losses intact; pitch loss directly targets prosody.

7. Tuning knobs and notes:

    - lambda_pitch: 0.1–0.3 is a good starting range. If intonation overwhelms timbre, lower it; if monotone, raise slightly.
    - F0 range in datamodule get_f0 is 50–1100 Hz by default; adjust if your dataset differs (e.g., 50–700 Hz for typical adult speech).
    - F0 quality depends on clean data; noisy segments will produce sparse masks and weaker gradients.

8. Expected outcomes:

    - Faster convergence on natural intonation and improved prosody stability.
    - Better phrasing/expressiveness with the same architecture capacity and vocoder.
