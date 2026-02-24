""" from https://github.com/jaywalnut310/glow-tts """

import numpy as np
import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
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


def normalize(data, mu, std):
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return (data - mu) / std


def denormalize(data, mu, std):
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu
