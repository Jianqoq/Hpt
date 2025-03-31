#pragma once
#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"
#include "../utils/make_vec.cuh"

template <typename Input, typename Output, typename Op, int vec_size>
__device__ __forceinline__ void unary_contiguous(Output *out, Input *in, int32_t n, Op op)
{
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    // {
    //     out[idx] = op(cast<Input, Output>(in[idx]));
    // }
    using InputVec = typename VectorTrait<Input, vec_size>::type;
    using OutputVec = typename VectorTrait<Output, vec_size>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // gridDim.x is already divided by vec_size
    int n_vec = n / vec_size;
    OutputVec out_vec;
    InputVec in_vec;
    for (int i = idx; i < n_vec; i += stride)
    {
        in_vec = VecMaker<Input>::make<vec_size>(in + i * vec_size);
        if constexpr (vec_size == 1)
            out_vec = op(cast<Input, Output>(in_vec));
        else if constexpr (vec_size > 1)
            out_vec.x = op(cast<Input, Output>(in_vec.x));
        if constexpr (vec_size >= 2)
            out_vec.y = op(cast<Input, Output>(in_vec.y));
        if constexpr (vec_size >= 3)
            out_vec.z = op(cast<Input, Output>(in_vec.z));
        if constexpr (vec_size >= 4)
            out_vec.w = op(cast<Input, Output>(in_vec.w));
        *reinterpret_cast<OutputVec *>(out + i * vec_size) = out_vec;
    }
    int remain_start = n_vec * vec_size;
    for (int i = remain_start + idx; i < n; i += stride)
    {
        out[i] = op(cast<Input, Output>(in[i]));
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
    __launch_bounds__(128) extern "C" __global__ void func_name##_contiguous(void *out, void *in, int32_t n)                                                              \
    {                                                                                                                                                                     \
        using Output = typename promote<input_type>::Output;                                                                                                              \
        unary_contiguous<input_type, Output, op<Output>, 4>(static_cast<Output *>(out), static_cast<input_type *>(in), n, op<Output>{});                                  \
    }                                                                                                                                                                     \
    extern "C" __global__ void func_name##_uncontiguous(void *out, void *in, int32_t size, FastDivmod *in_shape, int32_t *in_strides, int32_t in_ndim)                    \
    {                                                                                                                                                                     \
        using Output = typename promote<input_type>::Output;                                                                                                              \
        unary_uncontiguous<input_type, Output, op<Output>>(static_cast<Output *>(out), static_cast<input_type *>(in), size, in_shape, in_strides, in_ndim, op<Output>{}); \
    }
