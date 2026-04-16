""" from https://github.com/jaywalnut310/glow-tts """

import torch
import torch.nn.functional as F


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


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


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


def downsample(mu_y_fine):
    """
    Halves the time resolution of a mel spectrogram by averaging pairs of adjacent frames.
    If the original had a hop length of 128, the result will have a hop of 256.
    """
    mu_y = F.avg_pool1d(mu_y_fine, kernel_size=2, stride=2)

    # This does more averaging, and it could fix the harshness in speaker 4:
    # mu_y = F.avg_pool1d(mu_y_fine, kernel_size=3, stride=2, padding=1)

    return mu_y
