#pragma once
#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_contiguous(Output *out, const Input *in, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<Input, Output>(in[idx]));
    }
}

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_uncontiguous(
    Output *out,
    Input *in,
    int32_t size,
    FastDivmod *in_shape,
    int32_t *in_strides,
    int32_t in_ndim,
    Op op)
{
    UncontiguousIndexCalculator<Input> in_idx_calculator = UncontiguousIndexCalculator<Input>(in, in_shape, in_strides, in_ndim);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<Input, Output>(in_idx_calculator.get(idx)));
    }
}

#define DEFINE_UNARY_KERNEL(func_name, input_type, promote, op)                                                                                                           \
    extern "C" __global__ void func_name##_contiguous(void *out, void *in, int32_t n)                                                                                     \
    {                                                                                                                                                                     \
        using Output = typename promote<input_type>::Output;                                                                                                              \
        unary_contiguous<input_type, Output, op<Output>>(static_cast<Output *>(out), static_cast<const input_type *>(in), n, op<Output>{});                               \
    }                                                                                                                                                                     \
    extern "C" __global__ void func_name##_uncontiguous(void *out, void *in, int32_t size, FastDivmod *in_shape, int32_t *in_strides, int32_t in_ndim)                    \
    {                                                                                                                                                                     \
        using Output = typename promote<input_type>::Output;                                                                                                              \
        unary_uncontiguous<input_type, Output, op<Output>>(static_cast<Output *>(out), static_cast<input_type *>(in), size, in_shape, in_strides, in_ndim, op<Output>{}); \
    }
