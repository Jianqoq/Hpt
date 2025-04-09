#include "../utils/type_alias.cuh"


template <typename T, typename Intermediate>
__device__ __forceinline__ void pooling2d_forward(
    T *input, T *output,
    i32 batch_size, i32 channels,
    i32 input_height, i32 input_width,
    i32 output_height, i32 output_width,
    i32 kernel_h, i32 kernel_w,
    i32 stride_h, i32 stride_w,
    i32 padding_h, i32 padding_w,
    Intermediate *workspace)
{
    i32 global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
}
