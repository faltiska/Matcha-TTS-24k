"""
Everything related to perceptual weighting of the prior loss lives in this file: building the per-mel-bin
weight curve, and plotting it so the shape can be inspected visually rather than read from the code.

The theory behind this:
A Perceptually Motivated Loss Function for Speech Enhancement Based on Equal-Loudness Contours" (arXiv:2511.05945, 2025) 
Very recent paper doing almost exactly what your current implementation does — ISO 226 equal-loudness contours weighting 
a reconstruction loss for speech enhancement (not TTS).

You can use this script from CLI to generate a PNG file with the curve used for weighting the loss.
CLI usage (plots the curve for a given data config):
    python -m matcha.utils.perceptual_mel_weights
    python -m matcha.utils.perceptual_mel_weights -i configs/data/corpus-24k.yaml -o documentation/perceptual_mel_weights.png
"""
import argparse
import math
from pathlib import Path

import torch

DEFAULT_DATA_CONFIG = "configs/data/corpus-24k.yaml"
DEFAULT_OUTPUT_PATH = "documentation/perceptual_mel_weights.png"


def hz_to_htk_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def htk_mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def htk_mel_bin_center_frequencies(n_feats, f_min, f_max):
    """Frequencies (Hz) of each mel bin center, matching torchaudio MelSpectrogram(mel_scale='htk')."""
    mel_min = hz_to_htk_mel(f_min)
    mel_max = hz_to_htk_mel(f_max)
    # torchaudio places bin centers at n_feats+2 equally spaced mel points and drops the two edges,
    # so bin i's center is at mel_min + (i + 1) * mel_step, with mel_step spanning n_feats + 1 steps.
    mel_step = (mel_max - mel_min) / (n_feats + 1)
    bin_centers_mel = mel_min + torch.arange(1, n_feats + 1, dtype=torch.float64) * mel_step
    return torch.tensor([htk_mel_to_hz(m.item()) for m in bin_centers_mel], dtype=torch.float64)


def _interpolate_1d(query_points, known_x, known_y):
    """
    Linear interpolation of known_y (defined at strictly increasing known_x) at each query point.
    Query points outside the known range clamp to the nearest endpoint value.
    torch has no built-in 1-D interpolation, so this implements the standard piecewise-linear form.
    """
    right_index = torch.searchsorted(known_x, query_points).clamp(1, known_x.numel() - 1)
    left_index = right_index - 1

    left_x = known_x[left_index]
    right_x = known_x[right_index]
    left_y = known_y[left_index]
    right_y = known_y[right_index]

    slope = (right_y - left_y) / (right_x - left_x)
    interpolated = left_y + slope * (query_points - left_x)

    below_range = query_points <= known_x[0]
    above_range = query_points >= known_x[-1]
    interpolated = torch.where(below_range, known_y[0], interpolated)
    interpolated = torch.where(above_range, known_y[-1], interpolated)
    return interpolated


def build_perceptual_mel_weights(n_feats, f_min, f_max, min_weight=0.7, max_weight=1.3):
    """
    Builds a fixed, per-mel-bin weight vector that biases a mel-domain loss towards the frequency ranges
    where human hearing is most sensitive, without introducing a second competing loss term.

    The weighting is derived from the 40-phon equal-loudness contour (ISO 226), the same psychoacoustic
    curve used by the "Loud-loss" work and consistent with the perceptually weighted MR-STFT loss used in
    Parallel WaveGAN. The 40-phon level is representative of conversational speech. The contour gives, for
    each frequency, the sound pressure level (SPL) required to be perceived as equally loud as a 1kHz tone;
    a lower required SPL means the ear is more sensitive at that frequency. Following the Loud-loss
    formulation, each frequency's raw importance is the reference SPL at 1kHz divided by the SPL at that
    frequency, so the most sensitive frequencies (around 2.5-4kHz, where the contour dips lowest) get the
    largest importance and the extremes (very low and very high frequencies) get the smallest.

    The raw importance is then linearly compressed into the [min_weight, max_weight] range. This range is
    deliberately narrow (0.7-1.3 by default) rather than the strong contrast of the raw contour, so the
    loss is nudged towards perceptually important bins without starving the extremes, which would risk
    suppressing the high-frequency detail that carries timbre and speaker identity.

    Because the curve follows the actual contour, sensitivity genuinely falls off below roughly 500Hz as
    well as at the high end, rather than staying near-maximal all the way down to 0Hz.

    Mel bin center frequencies are computed using the HTK mel scale, matching the torchaudio
    MelSpectrogram(mel_scale="htk") filterbank used by the Vocos mel extractor, so the weights line up
    with the actual frequency each mel bin represents.

    Args:
        n_feats (int): number of mel bins.
        f_min (float): lowest frequency covered by the mel filterbank, in Hz.
        f_max (float): highest frequency covered by the mel filterbank, in Hz.
        min_weight (float): weight assigned to the least sensitive bins (the extremes).
        max_weight (float): weight assigned to the most sensitive bins.

    Returns:
        torch.Tensor of shape (n_feats,), one weight per mel bin.
    """
    bin_centers_hz = htk_mel_bin_center_frequencies(n_feats, f_min, f_max)

    # 40-phon equal-loudness contour (ISO 226): frequency in Hz and the SPL in dB required at that
    # frequency to match the loudness of a 1kHz tone. A lower SPL means the ear is more sensitive there.
    contour_hz = torch.tensor([
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
    ], dtype=torch.float64)
    contour_spl = torch.tensor([
        99.85, 93.94, 88.17, 82.63, 77.78, 73.08, 68.48, 64.37, 60.59, 56.70, 53.41, 50.40,
        47.58, 44.98, 43.05, 41.34, 40.06, 40.01, 41.82, 42.51, 39.23, 36.51, 35.61, 36.65,
        40.01, 45.83, 51.80, 56.70, 60.59,
        # last 2 coefficients were 54.28, 51.49, but I changed them to have the curve drop more at the end, since the 
        # human voice does not reach those frequencies; the ISO curve is about sound in general, not voice.
    ], dtype=torch.float64)

    # Interpolate the contour at each mel bin center. Equal-loudness contours are read on a log frequency
    # axis, so interpolate in log10(Hz). Bin centers below/above the tabulated range clamp to the ends.
    log_contour_hz = torch.log10(contour_hz)
    log_bin_centers_hz = torch.log10(torch.clamp(bin_centers_hz, min=contour_hz[0].item(), max=contour_hz[-1].item()))
    bin_spl = _interpolate_1d(log_bin_centers_hz, log_contour_hz, contour_spl)

    # Loud-loss importance: reference SPL at 1kHz divided by the SPL at this frequency. Higher where the
    # ear is more sensitive (lower required SPL).
    reference_spl_at_1khz = contour_spl[contour_hz == 1000].item()
    raw_importance = reference_spl_at_1khz / bin_spl

    # Linearly compress the raw importance into the gentle [min_weight, max_weight] range.
    min_importance = torch.min(raw_importance)
    max_importance = torch.max(raw_importance)
    normalized_importance = (raw_importance - min_importance) / (max_importance - min_importance)
    weights = min_weight + normalized_importance * (max_weight - min_weight)
    return weights.float()


def main():
    parser = argparse.ArgumentParser(description="Plot the perceptual mel-bin weighting curve for the prior loss")
    parser.add_argument("-i", "--input-config", default=DEFAULT_DATA_CONFIG, help="Data yaml with n_feats, f_min, f_max")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH, help="Where to save the plot")
    args = parser.parse_args()

    import yaml
    with open(args.input_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_feats = int(cfg["n_feats"])
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])

    weights = build_perceptual_mel_weights(n_feats, f_min, f_max)
    bin_centers_hz = htk_mel_bin_center_frequencies(n_feats, f_min, f_max)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(bin_centers_hz.numpy(), weights.numpy(), linewidth=2, color="#c05621")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="neutral (1.0)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Multiplier applied to this mel bin's loss term")
    plt.title("Prior loss weight actually applied per mel bin (ISO 226 40-phon derived)")
    plt.xlim(0, f_max)
    plt.ylim(0.6, 1.4)
    plt.xticks(range(0, int(f_max) + 1, 1000))
    plt.yticks([round(0.6 + 0.05 * i, 2) for i in range(17)])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved perceptual weight curve to {output_path}")


if __name__ == "__main__":
    main()
