#include "../reduce/block_reduce.cuh"
#include "../reduce/reduce_classes.cuh"
#include "../unary/unary_classes.cuh"
#include "../binary/binary_classes.cuh"
#include "../utils/index_calculator.cuh"
#include <stdio.h>

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/PersistentSoftmax.cuh
__device__ __forceinline__ int log2_ceil(int value)
{
    int log2_value = 0;
    while ((1 << log2_value) < value)
        ++log2_value;
    return log2_value;
}

// https://zhuanlan.zhihu.com/p/408474710
template <typename T>
__device__ __forceinline__ void welford_update(T *count, T *mean, T *m2, T current_val)
{
    *count += Prod<T>::identity();
    T delta = current_val - *mean;
    *mean += delta / *count;
    T delta2 = current_val - *mean;
    *m2 += delta * delta2;
}

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
template <typename T>
__device__ __forceinline__ void welford_update(T b_mean, T b_m2, T b_count, T *mean, T *m2, T *count)
{
    if (b_count == 0)
    {
        return;
    }
    T new_count = *count + b_count;
    T nb_over_n = Div<T, T>()(b_count, new_count);
    T delta = b_mean - *mean;
    *mean += delta * nb_over_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
    *count = new_count;
}

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
template <typename T, int thread_group_width = 32>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T *mean,
                                             T *m2, T *count)
{
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2)
    {
        T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
        T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
        T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
        welford_update(b_mean, b_m2, b_count, mean, m2, count);
    }
}

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
template <typename T, int thread_group_width = 32>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T *mean,
                                                T *m2, T *count)
{
    WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
    *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
    *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
    *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

// each warp compute a row of layernorm
template <typename T, typename Output, typename Intermediate, i32 log2_num_elements, i32 WarpSize, bool divisible>
__device__ __forceinline__ void layernorm_warp(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    Exp<Intermediate> exp;
    Div<Intermediate, Intermediate> div;
    Sqrt<Intermediate> sqrt;
    Rsqrt<Intermediate> rsqrt;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);
    T *input_ptr = input;
    T *output_ptr = output;

    constexpr i32 cs = 1 << log2_num_elements;
    constexpr i32 num_elements = (cs + WarpSize - 1) / WarpSize;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        Intermediate local_elements[num_elements];

        for (i32 c = 0; c < num_elements; c++)
        {
            i32 col = threadIdx.x + c * blockDim.x;
            if constexpr (!divisible)
            {
                if (col < cols)
                {
                    local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
                    welford_update(&count, &mean, &m2, local_elements[c]);
                }
            }
            else
            {
                local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
                welford_update(&count, &mean, &m2, local_elements[c]);
            }
        }
        WelfordWarpAllReduce(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);
#pragma unroll
        for (i32 c = 0; c < num_elements; c++)
        {
            i32 col = threadIdx.x + c * blockDim.x;
            if constexpr (!divisible)
            {
                if (col < cols)
                {
                    output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
                }
            }
            else
            {
                output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
            }
        }
    }
}

// each block compute a row of layernorm
template <typename T, typename Output, typename Intermediate, i32 BlockSize, i32 WarpSize = 32>
__device__ __forceinline__ void layernorm_block(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols,
    Intermediate *shared_inp)
{
    Exp<Intermediate> exp;
    Div<Intermediate, Intermediate> div;
    __shared__ Intermediate max_smem[WarpSize];
    __shared__ Intermediate shared_max;
    __shared__ Intermediate shared_sum;
    __shared__ Intermediate shared_sum_exp_x[WarpSize];

    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    T *input_ptr = input;
    Output *output_ptr = output;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        Intermediate max_val = TypeUtils<Intermediate>::limit_min();
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate inp = cast<T, Intermediate>(input_ptr[col]);
            shared_inp[col] = inp;
            max_val = Max<Intermediate>::combine(max_val, inp);
        }
        max_val = blockReduce<Intermediate, Max<Intermediate, WarpSize>, Max<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(max_val, max_smem);
        if (threadIdx.x == 0)
            shared_max = max_val;
        __syncthreads();
        Intermediate sum_exp_x = Sum<Intermediate>::identity();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate exp_x = exp(shared_inp[col] - shared_max);
            sum_exp_x = Sum<Intermediate>::combine(sum_exp_x, exp_x);
            shared_inp[col] = exp_x;
        }
        sum_exp_x = blockReduce<Intermediate, Sum<Intermediate>, Sum<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(sum_exp_x, shared_sum_exp_x);
        if (threadIdx.x == 0)
            shared_sum = sum_exp_x;
        __syncthreads();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate exp_x = shared_inp[col];
            output_ptr[col] = cast<Intermediate, Output>(div(exp_x, shared_sum));
        }
    }
}

// each block compute a row of softmax
template <typename T, typename Output, typename Intermediate, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ __forceinline__ void layernorm_block_large(
    T *input,
    Output *output,
    Intermediate eps,
    Output *buffer,
    i32 rows,
    i32 cols)
{
    __shared__ T max_smem[WarpSize];
    T max_val = TypeUtils<T>::limit_min();
    __shared__ T shared_max;
    __shared__ Output shared_sum;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    T *input_ptr = input;
    Output *output_ptr = output;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            max_val = Max<T>::combine(max_val, input_ptr[col]);
        }
        max_val = blockReduce<T, Max<T, WarpSize>, Max<T, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(max_val, max_smem);
        if (threadIdx.x == 0)
            shared_max = max_val;
        __syncthreads();
        Output sum_exp_x = Sum<Output>::identity();

        __shared__ Output shared_sum_exp_x[WarpSize];
        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Output exp_x = Exp<T>()(input_ptr[col] - shared_max);
            sum_exp_x = Sum<Output>::combine(sum_exp_x, exp_x);
            buffer[col] = exp_x;
        }
        sum_exp_x = blockReduce<Output, Sum<Output>, Sum<Output, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(sum_exp_x, shared_sum_exp_x);
        if (threadIdx.x == 0)
            shared_sum = sum_exp_x;
        __syncthreads();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Output exp_x = buffer[col];
            output_ptr[col] = Div<Output, Output>()(exp_x, shared_sum);
        }
    }
}

#define LayernormWarpContiguousCase(T, TOutput, TIntermediate, log2_num_elements, warp_size)                                \
    case log2_num_elements:                                                                                                 \
        if ((1 << log2_num_elements) % warp_size == 0)                                                                      \
        {                                                                                                                   \
            layernorm_warp<T, TOutput, TIntermediate, log2_num_elements, warp_size, true>(input, output, eps, rows, cols);  \
        }                                                                                                                   \
        else                                                                                                                \
        {                                                                                                                   \
            layernorm_warp<T, TOutput, TIntermediate, log2_num_elements, warp_size, false>(input, output, eps, rows, cols); \
        }                                                                                                                   \
        break;

#define DECLARE_LAYERNORM_KERNEL(T)                                                                                                                                            \
    using T##Output = FloatOutUnaryPromote<T>::Output;                                                                                                                         \
    using T##Intermediate = FloatOutUnaryPromote<T>::Intermediate;                                                                                                             \
    extern "C" __global__ void T##_layernorm_warp(T *input, T##Output *output, T##Output eps, i32 rows, i32 cols)                                                              \
    {                                                                                                                                                                          \
        switch (log2_ceil(cols))                                                                                                                                               \
        {                                                                                                                                                                      \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 10, 32);                                                                                                \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 9, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 8, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 7, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 6, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 5, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 4, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 3, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 2, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 1, 32);                                                                                                 \
            LayernormWarpContiguousCase(T, T##Output, T##Intermediate, 0, 32);                                                                                                 \
        default:                                                                                                                                                               \
            break;                                                                                                                                                             \
        }                                                                                                                                                                      \
    }                                                                                                                                                                          \
    extern "C" __global__ void T##_layernorm_block(T *input, T##Output *output, T##Output *gamma, T##Output *beta, T##Output eps, i32 rows, i32 cols)                          \
    {                                                                                                                                                                          \
        extern __shared__ T##Intermediate shared_inp_##T##_block[];                                                                                                            \
        layernorm_block<T, T##Output, T##Intermediate, 1024, 32>(input, output, eps, rows, cols, shared_inp_##T##_block + threadIdx.y * cols);                                 \
    }                                                                                                                                                                          \
    extern "C" __global__ void T##_layernorm_block_large(T *input, T##Output *output, T##Output *gamma, T##Output *beta, T##Output eps, T##Output *buffer, i32 rows, i32 cols) \
    {                                                                                                                                                                          \
        layernorm_block_large<T, T##Output, T##Intermediate, 1024, 32>(input, output, eps, buffer, rows, cols);                                                                \
    }

// DECLARE_LAYERNORM_KERNEL(bf16);
DECLARE_LAYERNORM_KERNEL(f16);
DECLARE_LAYERNORM_KERNEL(f32);
// DECLARE_LAYERNORM_KERNEL(f64);
