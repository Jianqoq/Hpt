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

// reference https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh
// each warp compute a row of softmax
template <typename T, typename Output, typename Intermediate, bool Contiguous, i32 log2_num_elements, i32 WarpSize, bool divisible, bool log_softmax = false>
__device__ __forceinline__ void softmax_warp(
    T *input,
    Output *output,
    i32 rows,
    i32 cols,
    FastDivmod *divmod = nullptr,
    i32 *strides = nullptr,
    i32 ndim = 1,
    i32 last_stride = 1,
    FastDivmod *out_divmod = nullptr,
    i32 *out_strides = nullptr,
    i32 out_ndim = 1,
    i32 out_last_stride = 1)
{
    Exp<Intermediate> exp;

    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    T *input_ptr = input;
    T *output_ptr = output;

    constexpr i32 cs = 1 << log2_num_elements;
    constexpr i32 num_elements = (cs + WarpSize - 1) / WarpSize;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        Intermediate max_val = TypeUtils<Intermediate>::limit_min();
        if constexpr (!Contiguous)
        {
            UncontiguousIndexCalculator<T> idx_calc(input, divmod, strides, ndim);
            UncontiguousIndexCalculator<T> out_idx_calc(output, out_divmod, out_strides, out_ndim);
            input_ptr = idx_calc.get_ptr(row * cols);
            output_ptr = out_idx_calc.get_ptr(row * cols);
        }
        else
        {
            input_ptr = input + row * cols;
            output_ptr = output + row * cols;
        }
        Intermediate local_elements[num_elements];
#pragma unroll
        for (i32 c = 0; c < num_elements; c++)
        {
            i32 col = threadIdx.x + c * blockDim.x;
            if constexpr (!divisible)
            {
                local_elements[c] = (col < cols) ? cast<T, Intermediate>(input_ptr[col * last_stride]) : TypeUtils<Intermediate>::limit_min();
            }
            else
            {
                local_elements[c] = cast<T, Intermediate>(input_ptr[col * last_stride]);
            }
            max_val = Max<Intermediate>::combine(max_val, local_elements[c]);
        }
        max_val = Max<Intermediate>::warp_reduce(max_val);
        max_val = __shfl_sync(0xffffffff, max_val, 0);
        Intermediate sum = Sum<Intermediate>::identity();
#pragma unroll
        for (i32 c = 0; c < num_elements; c++)
        {
            if constexpr (!divisible)
            {
                i32 col = threadIdx.x + c * blockDim.x;
                if constexpr (log_softmax)
                    local_elements[c] = (col < cols) ? local_elements[c] - max_val : Sum<Intermediate>::identity();
                else
                    local_elements[c] = (col < cols) ? exp(local_elements[c] - max_val) : Sum<Intermediate>::identity();
            }
            else
            {
                if constexpr (log_softmax)
                    local_elements[c] = local_elements[c] - max_val;
                else
                    local_elements[c] = exp(local_elements[c] - max_val);
            }
            if constexpr (log_softmax)
                sum = Sum<Intermediate>::combine(sum, exp(local_elements[c]));
            else
                sum = Sum<Intermediate>::combine(sum, local_elements[c]);
        }
        sum = Sum<Intermediate>::warp_reduce(sum);
        sum = __shfl_sync(0xffffffff, sum, 0);
        if constexpr (log_softmax)
            sum = Ln<Intermediate>()(sum);
        Div<Intermediate, Intermediate> div;
#pragma unroll
        for (i32 c = 0; c < num_elements; c++)
        {
            i32 col = threadIdx.x + c * blockDim.x;
            if constexpr (!divisible)
            {
                if constexpr (log_softmax)
                {
                    if (col < cols)
                        output_ptr[col * out_last_stride] = cast<Intermediate, Output>(local_elements[c] - sum);
                }
                else
                {
                    if (col < cols)
                        output_ptr[col * out_last_stride] = cast<Intermediate, Output>(div(local_elements[c], sum));
                }
            }
            else
            {
                if constexpr (log_softmax)
                    output_ptr[col * out_last_stride] = cast<Intermediate, Output>(local_elements[c] - sum);
                else
                    output_ptr[col * out_last_stride] = cast<Intermediate, Output>(div(local_elements[c], sum));
            }
        }
    }
}

// each block compute a row of softmax
template <typename T, typename Output, typename Intermediate, bool Contiguous, i32 BlockSize, i32 WarpSize = 32, bool log_softmax = false>
__device__ __forceinline__ void softmax_block(
    T *input,
    Output *output,
    i32 rows,
    i32 cols,
    Intermediate *shared_inp,
    FastDivmod *divmod = nullptr,
    i32 *strides = nullptr,
    i32 ndim = 1,
    i32 last_stride = 1,
    FastDivmod *out_divmod = nullptr,
    i32 *out_strides = nullptr,
    i32 out_ndim = 1,
    i32 out_last_stride = 1)
{
    Exp<Intermediate> exp;
    Div<Intermediate, Intermediate> div;
    __shared__ Intermediate max_smem[WarpSize];
    __shared__ Intermediate shared_max;
    __shared__ Intermediate shared_sum;
    __shared__ Intermediate shared_sum_exp[WarpSize];

    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    T *input_ptr = input;
    Output *output_ptr = output;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        Intermediate max_val = TypeUtils<Intermediate>::limit_min();
        if constexpr (!Contiguous)
        {
            UncontiguousIndexCalculator<T> idx_calc(input, divmod, strides, ndim);
            UncontiguousIndexCalculator<T> out_idx_calc(output, out_divmod, out_strides, out_ndim);
            input_ptr = idx_calc.get_ptr(row * cols);
            output_ptr = out_idx_calc.get_ptr(row * cols);
        }
        else
        {
            input_ptr = input + row * cols;
            output_ptr = output + row * cols;
        }
        for (i32 col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate inp = cast<T, Intermediate>(input_ptr[col * last_stride]);
            shared_inp[col] = inp;
            max_val = Max<Intermediate>::combine(max_val, inp);
        }
        max_val = blockReduce<Intermediate, Max<Intermediate, WarpSize>, Max<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(max_val, max_smem);
        if (threadIdx.x == 0)
            shared_max = max_val;
        __syncthreads();
        Intermediate sum = Sum<Intermediate>::identity();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate x;
            if constexpr (log_softmax)
                x = shared_inp[col] - shared_max;
            else
                x = exp(shared_inp[col] - shared_max);
            shared_inp[col] = x;
            if constexpr (log_softmax)
                sum = Sum<Intermediate>::combine(sum, exp(x));
            else
                sum = Sum<Intermediate>::combine(sum, x);
        }
        sum = blockReduce<Intermediate, Sum<Intermediate>, Sum<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(sum, shared_sum_exp);
        if (threadIdx.x == 0)
        {
            if constexpr (log_softmax)
                shared_sum = Ln<Intermediate>()(sum);
            else
                shared_sum = sum;
        }
        __syncthreads();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            Intermediate x = shared_inp[col];
            if constexpr (log_softmax)
                output_ptr[col * out_last_stride] = cast<Intermediate, Output>(x - shared_sum);
            else
                output_ptr[col * out_last_stride] = cast<Intermediate, Output>(div(x, shared_sum));
        }
    }
}

// each block compute a row of softmax
template <typename T, typename Output, typename Intermediate, bool Contiguous, uint32_t BlockSize, uint32_t WarpSize = 32, bool log_softmax = false>
__device__ __forceinline__ void softmax_block_large(
    T *input,
    Output *output,
    Intermediate *buffer,
    i32 rows,
    i32 cols,
    FastDivmod *divmod = nullptr,
    i32 *strides = nullptr,
    i32 ndim = 1,
    i32 last_stride = 1,
    FastDivmod *out_divmod = nullptr,
    i32 *out_strides = nullptr,
    i32 out_ndim = 1,
    i32 out_last_stride = 1)
{
    Exp<Intermediate> exp;
    Div<Intermediate, Intermediate> div;
    __shared__ Intermediate max_smem[WarpSize];
    __shared__ Intermediate shared_max;
    __shared__ Intermediate shared_sum;
    i32 row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    T *input_ptr = input;
    Output *output_ptr = output;
    for (i32 row = row_idx; row < rows; row += blockDim.y * gridDim.y)
    {
        Intermediate max_val = TypeUtils<Intermediate>::limit_min();
        if constexpr (!Contiguous)
        {
            UncontiguousIndexCalculator<T> idx_calc(input, divmod, strides, ndim);
            UncontiguousIndexCalculator<T> out_idx_calc(output, out_divmod, out_strides, out_ndim);
            input_ptr = idx_calc.get_ptr(row * cols);
            output_ptr = out_idx_calc.get_ptr(row * cols);
        }
        else
        {
            input_ptr = input + row * cols;
            output_ptr = output + row * cols;
        }
        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            max_val = Max<Intermediate>::combine(max_val, cast<T, Intermediate>(input_ptr[col * last_stride]));
        }
        max_val = blockReduce<Intermediate, Max<Intermediate, WarpSize>, Max<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(max_val, max_smem);
        if (threadIdx.x == 0)
            shared_max = max_val;
        __syncthreads();
        Intermediate sum = Sum<Intermediate>::identity();

        __shared__ Intermediate shared_sum_exp[WarpSize];
        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            if constexpr (log_softmax)
            {
                Intermediate exp_x = cast<T, Intermediate>(input_ptr[col * last_stride]) - shared_max;
                buffer[col] = exp_x;
                sum = Sum<Intermediate>::combine(sum, exp(exp_x));
            }
            else
            {
                Intermediate exp_x = exp(cast<T, Intermediate>(input_ptr[col * last_stride]) - shared_max);
                sum = Sum<Intermediate>::combine(sum, exp_x);
                buffer[col] = exp_x;
            }
        }
        sum = blockReduce<Intermediate, Sum<Intermediate>, Sum<Intermediate, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(sum, shared_sum_exp);
        if constexpr (log_softmax)
        {
            if (threadIdx.x == 0)
                shared_sum = Ln<Intermediate>()(sum);
        }
        else
        {
            if (threadIdx.x == 0)
                shared_sum = sum;
        }
        __syncthreads();

        for (int col = threadIdx.x; col < cols; col += blockDim.x)
        {
            if constexpr (log_softmax)
            {
                Intermediate exp_x = buffer[col];
                output_ptr[col * out_last_stride] = cast<Intermediate, Output>(exp_x - shared_sum);
            }
            else
            {
                Intermediate exp_x = buffer[col];
                output_ptr[col * out_last_stride] = cast<Intermediate, Output>(div(exp_x, shared_sum));
            }
        }
    }
}

#define SoftmaxWarpContiguousCase(T, TOutput, TIntermediate, log2_num_elements, warp_size)                                 \
    case log2_num_elements:                                                                                                \
        if ((1 << log2_num_elements) % warp_size == 0)                                                                     \
        {                                                                                                                  \
            softmax_warp<T, TOutput, TIntermediate, true, log2_num_elements, warp_size, true>(input, output, rows, cols);  \
        }                                                                                                                  \
        else                                                                                                               \
        {                                                                                                                  \
            softmax_warp<T, TOutput, TIntermediate, true, log2_num_elements, warp_size, false>(input, output, rows, cols); \
        }                                                                                                                  \
        break;

#define SoftmaxWarpUnContiguousCase(T, TOutput, TIntermediate, log2_num_elements, warp_size)                                                                                                                                        \
    case log2_num_elements:                                                                                                                                                                                                         \
        if ((1 << log2_num_elements) % warp_size == 0)                                                                                                                                                                              \
        {                                                                                                                                                                                                                           \
            softmax_warp<T, TOutput, TIntermediate, false, log2_num_elements, warp_size, true>(input, output, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]);  \
        }                                                                                                                                                                                                                           \
        else                                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                           \
            softmax_warp<T, TOutput, TIntermediate, false, log2_num_elements, warp_size, false>(input, output, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]); \
        }                                                                                                                                                                                                                           \
        break;

#define DECLARE_SOFTMAX_KERNEL(T)                                                                                                                                                                                                                                   \
    using T##Output = FloatOutUnaryPromote<T>::Output;                                                                                                                                                                                                              \
    using T##Intermediate = FloatOutUnaryPromote<T>::Intermediate;                                                                                                                                                                                                  \
    extern "C" __global__ void T##_softmax_warp(T *input, T##Output *output, i32 rows, i32 cols)                                                                                                                                                                    \
    {                                                                                                                                                                                                                                                               \
        switch (log2_ceil(cols))                                                                                                                                                                                                                                    \
        {                                                                                                                                                                                                                                                           \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 10, 32);                                                                                                                                                                                       \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 9, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 8, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 7, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 6, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 5, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 4, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 3, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 2, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 1, 32);                                                                                                                                                                                        \
            SoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 0, 32);                                                                                                                                                                                        \
        default:                                                                                                                                                                                                                                                    \
            break;                                                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                                                           \
    }                                                                                                                                                                                                                                                               \
    extern "C" __global__ void T##_softmax_warp_uncontiguous(T *input, T##Output *output, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                                                   \
    {                                                                                                                                                                                                                                                               \
        switch (log2_ceil(cols))                                                                                                                                                                                                                                    \
        {                                                                                                                                                                                                                                                           \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 10, 32);                                                                                                                                                                                     \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 9, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 8, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 7, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 6, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 5, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 4, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 3, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 2, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 1, 32);                                                                                                                                                                                      \
            SoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 0, 32);                                                                                                                                                                                      \
        default:                                                                                                                                                                                                                                                    \
            break;                                                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                                                           \
    }                                                                                                                                                                                                                                                               \
    extern "C" __global__ void T##_softmax_block(T *input, T##Output *output, i32 rows, i32 cols)                                                                                                                                                                   \
    {                                                                                                                                                                                                                                                               \
        extern __shared__ T##Intermediate shared_inp_##T##_block[];                                                                                                                                                                                                 \
        softmax_block<T, T##Output, T##Intermediate, true, 1024, 32>(input, output, rows, cols, shared_inp_##T##_block + threadIdx.y * cols);                                                                                                                       \
    }                                                                                                                                                                                                                                                               \
    extern "C" __global__ void T##_softmax_block_uncontiguous(T *input, T##Output *output, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                                                  \
    {                                                                                                                                                                                                                                                               \
        extern __shared__ T##Intermediate shared_inp_##T##_block_uncontiguous[];                                                                                                                                                                                    \
        softmax_block<T, T##Output, T##Intermediate, false, 1024, 32>(input, output, rows, cols, shared_inp_##T##_block_uncontiguous + threadIdx.y * cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]); \
    }                                                                                                                                                                                                                                                               \
    extern "C" __global__ void T##_softmax_block_large(T *input, T##Output *output, T##Intermediate *buffer, i32 rows, i32 cols)                                                                                                                                    \
    {                                                                                                                                                                                                                                                               \
        softmax_block_large<T, T##Output, T##Intermediate, true, 1024, 32>(input, output, buffer, rows, cols);                                                                                                                                                      \
    }                                                                                                                                                                                                                                                               \
    extern "C" __global__ void T##_softmax_block_large_uncontiguous(T *input, T##Output *output, T##Intermediate *buffer, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                   \
    {                                                                                                                                                                                                                                                               \
        softmax_block_large<T, T##Output, T##Intermediate, false, 1024, 32>(input, output, buffer, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]);                                             \
    }

// DECLARE_SOFTMAX_KERNEL(bf16);
DECLARE_SOFTMAX_KERNEL(f16);
DECLARE_SOFTMAX_KERNEL(f32);
DECLARE_SOFTMAX_KERNEL(f64);
// DECLARE_SOFTMAX_KERNEL(bool);
// DECLARE_SOFTMAX_KERNEL(i8);
// DECLARE_SOFTMAX_KERNEL(i16);
// DECLARE_SOFTMAX_KERNEL(i32);
// DECLARE_SOFTMAX_KERNEL(i64);
// DECLARE_SOFTMAX_KERNEL(u8);
// DECLARE_SOFTMAX_KERNEL(u16);
// DECLARE_SOFTMAX_KERNEL(u32);
// DECLARE_SOFTMAX_KERNEL(u64);

#define LogSoftmaxWarpContiguousCase(T, TOutput, TIntermediate, log2_num_elements, warp_size)                                    \
    case log2_num_elements:                                                                                                      \
        if ((1 << log2_num_elements) % warp_size == 0)                                                                           \
        {                                                                                                                        \
            softmax_warp<T, TOutput, TIntermediate, true, log2_num_elements, warp_size, true, true>(input, output, rows, cols);  \
        }                                                                                                                        \
        else                                                                                                                     \
        {                                                                                                                        \
            softmax_warp<T, TOutput, TIntermediate, true, log2_num_elements, warp_size, false, true>(input, output, rows, cols); \
        }                                                                                                                        \
        break;

#define LogSoftmaxWarpUnContiguousCase(T, TOutput, TIntermediate, log2_num_elements, warp_size)                                                                                                                                           \
    case log2_num_elements:                                                                                                                                                                                                               \
        if ((1 << log2_num_elements) % warp_size == 0)                                                                                                                                                                                    \
        {                                                                                                                                                                                                                                 \
            softmax_warp<T, TOutput, TIntermediate, false, log2_num_elements, warp_size, true, true>(input, output, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]);  \
        }                                                                                                                                                                                                                                 \
        else                                                                                                                                                                                                                              \
        {                                                                                                                                                                                                                                 \
            softmax_warp<T, TOutput, TIntermediate, false, log2_num_elements, warp_size, false, true>(input, output, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]); \
        }                                                                                                                                                                                                                                 \
        break;

#define DECLARE_LOG_SOFTMAX_KERNEL(T)                                                                                                                                                                                                                                     \
    using T##Output = FloatOutUnaryPromote<T>::Output;                                                                                                                                                                                                                    \
    using T##Intermediate = FloatOutUnaryPromote<T>::Intermediate;                                                                                                                                                                                                        \
    extern "C" __global__ void T##_logsoftmax_warp(T *input, T##Output *output, i32 rows, i32 cols)                                                                                                                                                                       \
    {                                                                                                                                                                                                                                                                     \
        switch (log2_ceil(cols))                                                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                                                                 \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 10, 32);                                                                                                                                                                                          \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 9, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 8, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 7, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 6, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 5, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 4, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 3, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 2, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 1, 32);                                                                                                                                                                                           \
            LogSoftmaxWarpContiguousCase(T, T##Output, T##Intermediate, 0, 32);                                                                                                                                                                                           \
        default:                                                                                                                                                                                                                                                          \
            break;                                                                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                                                                 \
    }                                                                                                                                                                                                                                                                     \
    extern "C" __global__ void T##_logsoftmax_warp_uncontiguous(T *input, T##Output *output, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                                                      \
    {                                                                                                                                                                                                                                                                     \
        switch (log2_ceil(cols))                                                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                                                                 \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 10, 32);                                                                                                                                                                                        \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 9, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 8, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 7, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 6, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 5, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 4, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 3, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 2, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 1, 32);                                                                                                                                                                                         \
            LogSoftmaxWarpUnContiguousCase(T, T##Output, T##Intermediate, 0, 32);                                                                                                                                                                                         \
        default:                                                                                                                                                                                                                                                          \
            break;                                                                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                                                                 \
    }                                                                                                                                                                                                                                                                     \
    extern "C" __global__ void T##_logsoftmax_block(T *input, T##Output *output, i32 rows, i32 cols)                                                                                                                                                                      \
    {                                                                                                                                                                                                                                                                     \
        extern __shared__ T##Intermediate shared_inp_##T##_block[];                                                                                                                                                                                                       \
        softmax_block<T, T##Output, T##Intermediate, true, 1024, 32, true>(input, output, rows, cols, shared_inp_##T##_block + threadIdx.y * cols);                                                                                                                       \
    }                                                                                                                                                                                                                                                                     \
    extern "C" __global__ void T##_logsoftmax_block_uncontiguous(T *input, T##Output *output, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                                                     \
    {                                                                                                                                                                                                                                                                     \
        extern __shared__ T##Intermediate shared_inp_##T##_block_uncontiguous[];                                                                                                                                                                                          \
        softmax_block<T, T##Output, T##Intermediate, false, 1024, 32, true>(input, output, rows, cols, shared_inp_##T##_block_uncontiguous + threadIdx.y * cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]); \
    }                                                                                                                                                                                                                                                                     \
    extern "C" __global__ void T##_logsoftmax_block_large(T *input, T##Output *output, T##Intermediate *buffer, i32 rows, i32 cols)                                                                                                                                       \
    {                                                                                                                                                                                                                                                                     \
        softmax_block_large<T, T##Output, T##Intermediate, true, 1024, 32, true>(input, output, buffer, rows, cols);                                                                                                                                                      \
    }                                                                                                                                                                                                                                                                     \
    extern "C" __global__ void T##_logsoftmax_block_large_uncontiguous(T *input, T##Output *output, T##Intermediate *buffer, i32 rows, i32 cols, FastDivmod *divmod, i32 *strides, i32 ndim, FastDivmod *out_divmod, i32 *out_strides, i32 out_ndim)                      \
    {                                                                                                                                                                                                                                                                     \
        softmax_block_large<T, T##Output, T##Intermediate, false, 1024, 32, true>(input, output, buffer, rows, cols, divmod, strides, ndim, strides[ndim - 1], out_divmod, out_strides, out_ndim, out_strides[out_ndim - 1]);                                             \
    }

// DECLARE_LOG_SOFTMAX_KERNEL(bf16);
DECLARE_LOG_SOFTMAX_KERNEL(f16);
DECLARE_LOG_SOFTMAX_KERNEL(f32);
DECLARE_LOG_SOFTMAX_KERNEL(f64);
// DECLARE_LOG_SOFTMAX_KERNEL(bool);
// DECLARE_LOG_SOFTMAX_KERNEL(i8);
// DECLARE_LOG_SOFTMAX_KERNEL(i16);
// DECLARE_LOG_SOFTMAX_KERNEL(i32);
// DECLARE_LOG_SOFTMAX_KERNEL(i64);
// DECLARE_LOG_SOFTMAX_KERNEL(u8);
// DECLARE_LOG_SOFTMAX_KERNEL(u16);
// DECLARE_LOG_SOFTMAX_KERNEL(u32);
// DECLARE_LOG_SOFTMAX_KERNEL(u64);