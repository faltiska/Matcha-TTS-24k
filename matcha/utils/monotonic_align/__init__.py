import numpy as np
import torch
from matcha.utils.monotonic_align.core import maximum_path_c
from torch.utils.cpp_extension import load_inline

def maximum_path_cpu(value, mask):
    """Cython optimised version.
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
    return torch.from_numpy(path).to(device=device, dtype=dtype)

def maximum_path_cpu_2(value, mask):
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

# ##################################################################################################
# # CUDA kernel code
# cuda_source = """
# #include <torch/extension.h>
# #include <cuda_runtime.h>
# 
# __global__ void maximum_path_kernel(
#     int* paths,
#     float* values,
#     const int* t_xs,
#     const int* t_ys,
#     int batch_size,
#     int max_t_x,
#     int max_t_y,
#     float max_neg_val
# ) {
#     int b = blockIdx.x * blockDim.x + threadIdx.x;
#     
#     if (b >= batch_size) return;
#     
#     int t_x = t_xs[b];
#     int t_y = t_ys[b];
#     
#     // Offset pointers for this batch
#     float* value = values + b * max_t_x * max_t_y;
#     int* path = paths + b * max_t_x * max_t_y;
#     
#     // Forward pass - compute cumulative values
#     for (int y = 0; y < t_y; y++) {
#         for (int x = max(0, t_x + y - t_y); x < min(t_x, y + 1); x++) {
#             float v_cur, v_prev;
#             
#             // Get v_cur
#             if (x == y) {
#                 v_cur = max_neg_val;
#             } else {
#                 v_cur = value[x * max_t_y + (y - 1)];
#             }
#             
#             // Get v_prev
#             if (x == 0) {
#                 if (y == 0) {
#                     v_prev = 0.0f;
#                 } else {
#                     v_prev = max_neg_val;
#                 }
#             } else {
#                 v_prev = value[(x - 1) * max_t_y + (y - 1)];
#             }
#             
#             value[x * max_t_y + y] = fmaxf(v_cur, v_prev) + value[x * max_t_y + y];
#         }
#     }
#     
#     // Backward pass - find optimal path
#     int index = t_x - 1;
#     for (int y = t_y - 1; y >= 0; y--) {
#         path[index * max_t_y + y] = 1;
#         if (index != 0 && (index == y || value[index * max_t_y + (y - 1)] < value[(index - 1) * max_t_y + (y - 1)])) {
#             index = index - 1;
#         }
#     }
# }
# 
# torch::Tensor maximum_path_cuda(
#     torch::Tensor paths,
#     torch::Tensor values,
#     torch::Tensor t_xs,
#     torch::Tensor t_ys,
#     float max_neg_val
# ) {
#     const int batch_size = values.size(0);
#     const int max_t_x = values.size(1);
#     const int max_t_y = values.size(2);
#     
#     const int threads = 256;
#     const int blocks = (batch_size + threads - 1) / threads;
#     
#     maximum_path_kernel<<<blocks, threads>>>(
#         paths.data_ptr<int>(),
#         values.data_ptr<float>(),
#         t_xs.data_ptr<int>(),
#         t_ys.data_ptr<int>(),
#         batch_size,
#         max_t_x,
#         max_t_y,
#         max_neg_val
#     );
#     
#     return paths;
# }
# """
# 
# cpp_source = """
# torch::Tensor maximum_path_cuda(
#     torch::Tensor paths,
#     torch::Tensor values,
#     torch::Tensor t_xs,
#     torch::Tensor t_ys,
#     float max_neg_val
# );
# """
# 
# # Compile the extension
# maximum_path_cuda_module = load_inline(
#     name='maximum_path_cuda',
#     cpp_sources=cpp_source,
#     cuda_sources=cuda_source,
#     functions=['maximum_path_cuda'],
#     verbose=True,
#     extra_cuda_cflags=['-O3']
# )
# 
# def maximum_path_gpu(value, mask, max_neg_val=-1e9):
#     """
#     GPU version of maximum_path using CUDA.
#     
#     Args:
#         value: torch.Tensor of shape (batch, max_t_x, max_t_y) - input values
#         mask: torch.Tensor of shape (batch, max_t_x, max_t_y) or (batch, 1, max_t_x, max_t_y)
#         max_neg_val: float - large negative value for masking
#     
#     Returns:
#         paths: torch.Tensor of shape (batch, max_t_x, max_t_y) - binary path matrix
#     """
#     assert value.is_cuda, "Input must be on CUDA device"
#     
#     dtype = value.dtype
#     if mask.dim() == 4:
#         mask = mask.squeeze(1)
#     
#     value = value * mask
#     if value.dtype != torch.float32:
#         value = value.float()
#     
#     t_xs = mask.sum(1)[:, 0].int()
#     t_ys = mask.sum(2)[:, 0].int()
#     
#     paths = torch.zeros_like(value, dtype=torch.int32)
#     value = value.contiguous()
#     t_xs = t_xs.contiguous()
#     t_ys = t_ys.contiguous()
#     
#     maximum_path_cuda_module.maximum_path_cuda(paths, value, t_xs, t_ys, max_neg_val)
#     
#     if dtype != torch.int32:
#         paths = paths.to(dtype)
#     
#     return paths
