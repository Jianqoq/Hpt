#include "../reduce/block_reduce.cuh"
#include "../reduce/reduce_classes.cuh"
#include "../unary/unary_classes.cuh"
#include "../binary/binary_classes.cuh"
#include "../utils/index_calculator.cuh"
#include "../utils/make_vec.cuh"
#include <stdio.h>

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/PersistentSoftmax.cuh
__device__ __forceinline__ int log2_ceil(int value)
{
    int log2_value = 0;
    while ((1 << log2_value) < value)
        ++log2_value;
    return log2_value;
}

__device__ __forceinline__ int ceil_div(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

__device__ __forceinline__ int prev_multiple_of(int value, int multiple)
{
    return (value / multiple) * multiple;
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

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
template <typename T, int kWarpSize>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T *result_mean, T *result_m2, T *result_count)
{
    __shared__ T mean_shared[kWarpSize];
    __shared__ T m2_shared[kWarpSize];
    __shared__ T count_shared[kWarpSize];
    __shared__ T mean_result_broadcast;
    __shared__ T m2_result_broadcast;
    __shared__ T count_result_broadcast;
    const int lid = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    T warp_mean = Sum<T>::identity();
    T warp_m2 = Sum<T>::identity();
    T warp_count = Sum<T>::identity();
    WelfordWarpReduce<T, kWarpSize>(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
    __syncthreads();
    if (lid == 0)
    {
        mean_shared[wid] = warp_mean;
        m2_shared[wid] = warp_m2;
        count_shared[wid] = warp_count;
    }
    __syncthreads();
    if (wid == 0)
    {
        if (threadIdx.x < blockDim.x / kWarpSize)
        {
            warp_mean = mean_shared[lid];
            warp_m2 = m2_shared[lid];
            warp_count = count_shared[lid];
        }
        else
        {
            warp_mean = Sum<T>::identity();
            warp_m2 = Sum<T>::identity();
            warp_count = Sum<T>::identity();
        }
        __syncwarp();
        T block_mean = Sum<T>::identity();
        T block_m2 = Sum<T>::identity();
        T block_count = Sum<T>::identity();
        WelfordWarpReduce<T, kWarpSize>(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
        if (lid == 0)
        {
            mean_result_broadcast = block_mean;
            m2_result_broadcast = block_m2;
            count_result_broadcast = block_count;
        }
    }
    __syncthreads();
    *result_mean = mean_result_broadcast;
    *result_m2 = m2_result_broadcast;
    *result_count = count_result_broadcast;
}

// each warp compute a row of layernorm, cols is multiple of WarpSize
template <typename T, typename Output, typename Intermediate, i32 num_elements, i32 WarpSize, bool has_rem = false>
__device__ __forceinline__ void layernorm_warp_multiple_of_warp_size(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    using InpVec = VecMaker<T>::Vec4;
    using IntermediateVec = VecMaker<Intermediate>::Vec4;

    Div<Intermediate, Intermediate> div;
    Rsqrt<Intermediate> rsqrt;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);
    T *input_ptr = input;
    T *output_ptr = output;

    constexpr i32 vec_size = 4;

    constexpr i32 n_vec = num_elements / vec_size;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        Intermediate local_elements[num_elements];

        for (i32 c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            InpVec inp_vec = VecMaker<T>::make<vec_size>(input_ptr + col);
            local_elements[c * vec_size] = cast<T, Intermediate>(inp_vec.x);
            local_elements[c * vec_size + 1] = cast<T, Intermediate>(inp_vec.y);
            local_elements[c * vec_size + 2] = cast<T, Intermediate>(inp_vec.z);
            local_elements[c * vec_size + 3] = cast<T, Intermediate>(inp_vec.w);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 1]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 2]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 3]);
        }
        if constexpr (has_rem)
        {
            constexpr i32 rem = num_elements % vec_size;
            for (int c = num_elements - rem; c < num_elements; c++)
            {
                i32 col = threadIdx.x + c * blockDim.x;
                local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
                welford_update(&count, &mean, &m2, local_elements[c]);
            }
        }
        WelfordWarpAllReduce(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);

        for (i32 c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c * vec_size] - mean) * inv_std);
            output_ptr[col + 1] = cast<Intermediate, Output>((local_elements[c * vec_size + 1] - mean) * inv_std);
            output_ptr[col + 2] = cast<Intermediate, Output>((local_elements[c * vec_size + 2] - mean) * inv_std);
            output_ptr[col + 3] = cast<Intermediate, Output>((local_elements[c * vec_size + 3] - mean) * inv_std);
        }
        if constexpr (has_rem)
        {
            constexpr i32 rem = num_elements % vec_size;
            for (int c = num_elements - rem; c < num_elements; c++)
            {
                i32 col = threadIdx.x + c * blockDim.x;
                output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
            }
        }
    }
}

// each warp compute a row of layernorm
template <typename T, typename Output, typename Intermediate, i32 num_elements, i32 WarpSize>
__device__ __forceinline__ void layernorm_warp(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    using InpVec = VecMaker<T>::Vec3;
    using IntermediateVec = VecMaker<Intermediate>::Vec3;
    constexpr i32 vec_size = 3;

    Div<Intermediate, Intermediate> div;
    Rsqrt<Intermediate> rsqrt;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);
    T *input_ptr = input;
    T *output_ptr = output;

    i32 n_elements = prev_multiple_of(cols, WarpSize) / WarpSize;
    i32 n_vec = n_elements / vec_size;
    i32 rem = n_vec * vec_size;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        Intermediate local_elements[num_elements];

        i32 c;
        for (c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            InpVec inp_vec = VecMaker<T>::make<vec_size>(input_ptr + col);
            local_elements[c * vec_size] = cast<T, Intermediate>(inp_vec.x);
            local_elements[c * vec_size + 1] = cast<T, Intermediate>(inp_vec.y);
            local_elements[c * vec_size + 2] = cast<T, Intermediate>(inp_vec.z);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 1]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 2]);
        }
        for (int col = rem + threadIdx.x; col < cols; col += blockDim.x, c++)
        {
            local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
            welford_update(&count, &mean, &m2, local_elements[c]);
        }
        WelfordWarpAllReduce(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);

        for (c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c * vec_size] - mean) * inv_std);
            output_ptr[col + 1] = cast<Intermediate, Output>((local_elements[c * vec_size + 1] - mean) * inv_std);
            output_ptr[col + 2] = cast<Intermediate, Output>((local_elements[c * vec_size + 2] - mean) * inv_std);
        }
        for (int col = rem + threadIdx.x; col < cols; col += blockDim.x, c++)
        {
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
        }
    }
}

// each warp compute a row of layernorm, cols is multiple of WarpSize
template <typename T, typename Output, typename Intermediate, i32 num_elements, i32 WarpSize, bool has_rem = false>
__device__ __forceinline__ void layernorm_block_multiple_of_128(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    using InpVec = VecMaker<T>::Vec4;
    using IntermediateVec = VecMaker<Intermediate>::Vec4;

    Div<Intermediate, Intermediate> div;
    Rsqrt<Intermediate> rsqrt;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);
    T *input_ptr = input;
    T *output_ptr = output;

    constexpr i32 vec_size = 4;

    constexpr i32 n_vec = num_elements / vec_size;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        Intermediate local_elements[num_elements];

        for (i32 c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            InpVec inp_vec = VecMaker<T>::make<vec_size>(input_ptr + col);
            local_elements[c * vec_size] = cast<T, Intermediate>(inp_vec.x);
            local_elements[c * vec_size + 1] = cast<T, Intermediate>(inp_vec.y);
            local_elements[c * vec_size + 2] = cast<T, Intermediate>(inp_vec.z);
            local_elements[c * vec_size + 3] = cast<T, Intermediate>(inp_vec.w);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 1]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 2]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 3]);
        }
        if constexpr (has_rem)
        {
            constexpr i32 rem = num_elements % vec_size;
            for (int c = num_elements - rem; c < num_elements; c++)
            {
                i32 col = threadIdx.x + c * blockDim.x;
                local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
                welford_update(&count, &mean, &m2, local_elements[c]);
            }
        }
        WelfordBlockAllReduce<Intermediate, WarpSize>(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);

        for (i32 c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c * vec_size] - mean) * inv_std);
            output_ptr[col + 1] = cast<Intermediate, Output>((local_elements[c * vec_size + 1] - mean) * inv_std);
            output_ptr[col + 2] = cast<Intermediate, Output>((local_elements[c * vec_size + 2] - mean) * inv_std);
            output_ptr[col + 3] = cast<Intermediate, Output>((local_elements[c * vec_size + 3] - mean) * inv_std);
        }
        if constexpr (has_rem)
        {
            constexpr i32 rem = num_elements % vec_size;
            for (int c = num_elements - rem; c < num_elements; c++)
            {
                i32 col = threadIdx.x + c * blockDim.x;
                output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
            }
        }
    }
}

// each warp compute a row of layernorm
template <typename T, typename Output, typename Intermediate, i32 num_elements, i32 WarpSize>
__device__ __forceinline__ void layernorm_block_128(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    using InpVec = VecMaker<T>::Vec3;
    using IntermediateVec = VecMaker<Intermediate>::Vec3;
    constexpr i32 vec_size = 3;

    Div<Intermediate, Intermediate> div;
    Rsqrt<Intermediate> rsqrt;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);
    T *input_ptr = input;
    T *output_ptr = output;

    i32 n_elements = prev_multiple_of(cols, 128) / 128;
    i32 n_vec = n_elements / vec_size;
    i32 rem = n_vec * vec_size;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        Intermediate local_elements[num_elements];

        i32 c;
        for (c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            InpVec inp_vec = VecMaker<T>::make<vec_size>(input_ptr + col);
            local_elements[c * vec_size] = cast<T, Intermediate>(inp_vec.x);
            local_elements[c * vec_size + 1] = cast<T, Intermediate>(inp_vec.y);
            local_elements[c * vec_size + 2] = cast<T, Intermediate>(inp_vec.z);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 1]);
            welford_update(&count, &mean, &m2, local_elements[c * vec_size + 2]);
        }
        for (int col = rem + threadIdx.x; col < cols; col += blockDim.x, c++)
        {
            local_elements[c] = cast<T, Intermediate>(input_ptr[col]);
            welford_update(&count, &mean, &m2, local_elements[c]);
        }
        WelfordBlockAllReduce<Intermediate, WarpSize>(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);

        for (c = 0; c < n_vec; c++)
        {
            i32 col = (threadIdx.x + c * blockDim.x) * vec_size;
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c * vec_size] - mean) * inv_std);
            output_ptr[col + 1] = cast<Intermediate, Output>((local_elements[c * vec_size + 1] - mean) * inv_std);
            output_ptr[col + 2] = cast<Intermediate, Output>((local_elements[c * vec_size + 2] - mean) * inv_std);
        }
        for (int col = rem + threadIdx.x; col < cols; col += blockDim.x, c++)
        {
            output_ptr[col] = cast<Intermediate, Output>((local_elements[c] - mean) * inv_std);
        }
    }
}

// each block compute a row of layernorm
template <typename T, typename Output, typename Intermediate, i32 WarpSize = 32>
__device__ __forceinline__ void layernorm_block_large(
    T *input,
    Output *output,
    Output eps,
    i32 rows,
    i32 cols)
{
    Div<Intermediate, Intermediate> div;
    Rsqrt<Intermediate> rsqrt;
    Intermediate eps_intermediate = cast<Output, Intermediate>(eps);

    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    T *input_ptr = input;
    Output *output_ptr = output;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        input_ptr = input + row * cols;
        output_ptr = output + row * cols;
        Intermediate mean = Sum<Intermediate>::identity();
        Intermediate count = Sum<Intermediate>::identity();
        Intermediate m2 = Sum<Intermediate>::identity();
        for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate inp = cast<T, Intermediate>(input_ptr[col]);
            welford_update(&count, &mean, &m2, inp);
        }
        WelfordBlockAllReduce<Intermediate, WarpSize>(mean, m2, count, &mean, &m2, &count);

        Intermediate variance = div(m2, count);
        Intermediate inv_std = rsqrt(variance + eps_intermediate);

        for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate inp = cast<T, Intermediate>(input_ptr[col]);
            output_ptr[col] = cast<Intermediate, Output>((inp - mean) * inv_std);
        }
    }
}

#define LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, TOutput, TIntermediate, num, warp_size)                                                        \
    case num:                                                                                                                              \
        if (cols % 32 == 0)                                                                                                                \
        {                                                                                                                                  \
            layernorm_warp_multiple_of_warp_size<T, TOutput, TIntermediate, num, warp_size, num % 4 == 0>(input, output, eps, rows, cols); \
        }                                                                                                                                  \
        else                                                                                                                               \
        {                                                                                                                                  \
            layernorm_warp<T, TOutput, TIntermediate, num, warp_size>(input, output, eps, rows, cols);                                     \
        }                                                                                                                                  \
        break;

#define LAYER_NORM_MULTIPLE_OF_128(T, TOutput, TIntermediate, num, warp_size)                                                         \
    case num:                                                                                                                         \
        if (cols % 128 == 0)                                                                                                          \
        {                                                                                                                             \
            layernorm_block_multiple_of_128<T, TOutput, TIntermediate, num, warp_size, num % 4 == 0>(input, output, eps, rows, cols); \
        }                                                                                                                             \
        else                                                                                                                          \
        {                                                                                                                             \
            layernorm_block_128<T, TOutput, TIntermediate, num, warp_size>(input, output, eps, rows, cols);                           \
        }                                                                                                                             \
        break;

#define DECLARE_LAYERNORM_KERNEL(T)                                                                                      \
    using T##Output = FloatOutUnaryPromote<T>::Output;                                                                   \
    using T##Intermediate = FloatOutUnaryPromote<T>::Intermediate;                                                       \
    extern "C" __global__ void T##_layernorm_warp(T *input, T##Output *output, T##Output eps, i32 rows, i32 cols)        \
    {                                                                                                                    \
        switch (ceil_div(cols, 32))                                                                                      \
        {                                                                                                                \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 1, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 2, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 3, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 4, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 5, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 6, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 7, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 8, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 9, 32)                                       \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 10, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 11, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 12, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 13, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 14, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 15, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 16, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 17, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 18, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 19, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 20, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 21, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 22, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 23, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 24, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 25, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 26, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 27, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 28, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 29, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 30, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 31, 32)                                      \
            LAYER_NORM_MULTIPLE_OF_WARP_SIZE(T, T##Output, T##Intermediate, 32, 32)                                      \
        default:                                                                                                         \
            break;                                                                                                       \
        }                                                                                                                \
    }                                                                                                                    \
    extern "C" __global__ void T##_layernorm_block(T *input, T##Output *output, T##Output eps, i32 rows, i32 cols)       \
    {                                                                                                                    \
        switch (ceil_div(cols, 128))                                                                                     \
        {                                                                                                                \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 1, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 2, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 3, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 4, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 5, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 6, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 7, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 8, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 9, 32)                                             \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 10, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 11, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 12, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 13, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 14, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 15, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 16, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 17, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 18, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 19, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 20, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 21, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 22, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 23, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 24, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 25, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 26, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 27, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 28, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 29, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 30, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 31, 32)                                            \
            LAYER_NORM_MULTIPLE_OF_128(T, T##Output, T##Intermediate, 32, 32)                                            \
        default:                                                                                                         \
            break;                                                                                                       \
        }                                                                                                                \
    }                                                                                                                    \
    extern "C" __global__ void T##_layernorm_block_large(T *input, T##Output *output, T##Output eps, i32 rows, i32 cols) \
    {                                                                                                                    \
        layernorm_block_large<T, T##Output, T##Intermediate, 32>(input, output, eps, rows, cols);                        \
    }

DECLARE_LAYERNORM_KERNEL(bf16);
DECLARE_LAYERNORM_KERNEL(f16);
DECLARE_LAYERNORM_KERNEL(f32);
DECLARE_LAYERNORM_KERNEL(f64);
