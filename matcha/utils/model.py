""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F

LOG_DURATION_OFFSET = 4


def sequence_mask(length, max_length):
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

# num_downsamplings_in_unet must be len(channels) - 1. 
# See channels configuration in decoder/default.yaml
# If it is [256, 256], num_downsamplings_in_unet should be 1
# If it is [256, 256, 256], num_downsamplings_in_unet should be 2
def fix_len_compatibility(length, num_downsamplings_in_unet=1):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
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
    worse. See documentation/components.md, Mel Analysis Window.

    avg_pool1d counts the zero padding, so the first and last frames average against silence, which
    is fine because every recording starts and ends with silence.
    """
    return F.avg_pool1d(mu_y_fine, kernel_size=3, stride=2, padding=1)
