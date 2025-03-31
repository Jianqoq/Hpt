#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"

#include "unary_classes.cuh"

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_contiguous(Output *out, const Input *in, Output alpha, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<Input, Output>(in[idx]), alpha);
    }
}

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_uncontiguous(
    Output *out,
    Input *in,
    Output alpha,
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
        out[idx] = op(cast<Input, Output>(in_idx_calculator.get(idx)), alpha);
    }
}

#define DEFINE_UNARY_KERNEL(func_name, input_type, promote, op)                                                                                                                                                                    \
    using func_name##Output = typename promote<input_type>::Output;                                                                                                                                                                \
    extern "C" __global__ void func_name##_contiguous(void *out, void *in, func_name##Output alpha, int32_t n)                                                                                                                     \
    {                                                                                                                                                                                                                              \
        unary_contiguous<input_type, func_name##Output, op<func_name##Output>>(static_cast<func_name##Output *>(out), static_cast<const input_type *>(in), alpha, n, op<func_name##Output>{});                                     \
    }                                                                                                                                                                                                                              \
    extern "C" __global__ void func_name##_uncontiguous(void *out, void *in, func_name##Output alpha, int32_t size, FastDivmod *in_shape, int32_t *in_strides, int32_t in_ndim)                                                    \
    {                                                                                                                                                                                                                              \
        unary_uncontiguous<input_type, func_name##Output, op<func_name##Output>>(static_cast<func_name##Output *>(out), static_cast<input_type *>(in), alpha, size, in_shape, in_strides, in_ndim, op<func_name##Output>{}); \
    }

__device__ __forceinline__ f32 elu(f32 a, f32 alpha)
{
    return fmaxf(0.0f, a) + alpha * fminf((expf(a) - 1.0f), 0.0f);
}

__device__ __forceinline__ f64 elu(f64 a, f64 alpha)
{
    return fmax(0.0, a) + alpha * fmin((exp(a) - 1.0), 0.0);
}

__device__ __forceinline__ f16 elu(f16 a, f16 alpha)
{
    f32 res = elu(cast<f16, f32>(a), cast<f16, f32>(alpha));
    return cast<f32, f16>(res);
}

__device__ __forceinline__ bf16 elu(bf16 a, bf16 alpha)
{
    f32 res = elu(cast<bf16, f32>(a), cast<bf16, f32>(alpha));
    return cast<f32, bf16>(res);
}

DEFINE_UNARY_KERNEL(elu_f16, f16, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_bf16, bf16, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_f32, f32, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_f64, f64, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_bool, bool, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_i8, i8, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_i16, i16, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_i32, i32, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_i64, i64, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_u8, u8, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_u16, u16, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_u32, u32, FloatOutUnaryPromote, Elu);
DEFINE_UNARY_KERNEL(elu_u64, u64, FloatOutUnaryPromote, Elu);
