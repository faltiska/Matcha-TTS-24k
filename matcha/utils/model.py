""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F

LOG_DURATION_OFFSET = 4


def sequence_mask(length, max_length):
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def fix_len_compatibility(length, num_halvings=1):
    """
    Round `length` up to a multiple of 2 ** num_halvings, so that the UNet's skip connections
    align: Upsample1D doubles exactly, so an odd length would come back one frame longer than
    the skip it has to be concatenated with.

    num_halvings counts how many times the length gets halved, which depends on the resolution the caller is at:
     - len(channels) - 1     for a standard resolution mel, halved once per UNet downsampling
     - len(channels)         for a fine resolution mel, halved once more by downsample() before the UNet sees it

    See channels configuration in decoder/default.yaml.
    If it is [256, 256], pass 1 for a standard mel and 2 for a fine mel.
    If it is [256, 256, 256], pass 2 for a standard mel and 3 for a fine mel.
    """
    factor = torch.scalar_tensor(2).pow(num_halvings)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length

def generate_path(duration, mask):
    """
    Build an attention path from phoneme durations.

    Args:
        duration: (batch, t_x) phoneme durations. Must be natural numbers.
        mask: (batch, t_x, t_y) attention mask.
    Returns:
        path: (batch, t_x, t_y) binary alignment map.
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration.long(), 1)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    return path * mask


def normalize(data, mean, std):
    """
    Mean and Std are corpus-wide statistics, that should be precalculated before training.
    Using this normalization method allows us to invert it at inference time without knowing the original data.
    All other normalization methods would depend on audio properties (min, max, norm).
    """
    return (data - mean) / std


def denormalize(data, mean, std):
    """Inverse of normalize()"""
    return data * std + mean


def box_downsample(mu_y_fine):
    """
    Halves the time resolution of a mel spectrogram using an [1, 1, 1] / 3 filter.
    If the original had a hop length of 128, the result will have a hop of 256.

    The result is a bit blurred, and that is on purpose. Since both mel resolutions are extracted with
    the same analysis window, mu_y_fine[:, :, ::2] would already reproduce the standard mel exactly, so
    the smoothing is the only thing this function actually adds. The Decoder learns a correction from
    this mel toward the ground truth, and it copes better with a large consistent error than with a
    smaller one that varies frame to frame. A sharper [1, 2, 1] / 4 filter was tried in v22 and measured
    worse on the MCD test. See documentation/components.md, Mel Analysis Window.

    avg_pool1d counts the zero padding, so the first and last frames average against silence, which
    is fine because every recording starts and ends with silence.
    """
    return F.avg_pool1d(mu_y_fine, kernel_size=3, stride=2, padding=1)


def triangular_downsample(mu_y_fine):
    """
    Halves the time resolution of a mel spectrogram using a [1, 2, 1] / 4 filter.

    The filter is centred on each retained even-indexed frame. One zero frame is
    added at both boundaries before filtering, so the edges are weighted against
    silence just as they are in ``box_downsample``. This matches corpora whose
    recordings begin and end with silence.
    """
    padded = F.pad(mu_y_fine, (1, 1))
    return (padded[..., :-2:2] + 2 * padded[..., 1:-1:2] + padded[..., 2::2]) * 0.25


# Maps the `downsampler` hyperparameter to an implementation. Set it in configs/model/matcha.yaml and
# override it per experiment. It is saved with the checkpoint hyperparameters, so inference picks the
# same filter the model was trained with.
DOWNSAMPLERS = {
    "box": box_downsample,
    "triangular": triangular_downsample,
}

DEFAULT_DOWNSAMPLER = "box"


def get_downsampler(name):
    """
    Return the mel downsampler registered under `name`.

    An unknown name raises instead of falling back to a default: silently training or synthesising
    with the wrong filter would be far harder to notice than a failure at startup.
    """
    try:
        return DOWNSAMPLERS[name]
    except KeyError:
        raise ValueError(f"Unknown downsampler {name!r}. Available: {sorted(DOWNSAMPLERS)}") from None
