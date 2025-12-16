import numpy as np
import torch
from matcha.utils.monotonic_align.core import maximum_path_c

def maximum_path_cpu(value, mask):
    """Cython optimised version - CUDA graph compatible.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    device = value.device
    dtype = value.dtype

    value_masked = (value * mask).detach().cpu().to(torch.float32)
    value_np = np.ascontiguousarray(value_masked.numpy())
    mask_np = np.ascontiguousarray(mask.detach().cpu().numpy())
    path = np.zeros_like(value_np, dtype=np.int32)
    t_x_max = mask_np.sum(axis=1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(axis=2)[:, 0].astype(np.int32)

    maximum_path_c(path, value_np, t_x_max, t_y_max)

    # This creates the tensor on the correct device first, then copies the numpy data into it.
    # It avoids the CPU→GPU transfer that breaks CUDA graphs.
    # The copy_() operation is graph-compatible, facilitating model compilation.
    path_cpu = torch.from_numpy(path)
    path_tensor = torch.empty(path.shape, dtype=torch.int32, device=device)
    path_tensor.copy_(path_cpu)
    return path_tensor.to(dtype=dtype)