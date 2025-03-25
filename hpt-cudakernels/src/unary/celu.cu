#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_contiguous(Output *out, const Input *in, Output scale, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<Input, Output>(in[idx]), scale);
    }
}

template <typename Input, typename Output, typename Op>
__device__ __forceinline__ void unary_uncontiguous(
    Output *out,
    Input *in,
    Output scale,
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
        out[idx] = op(cast<Input, Output>(in_idx_calculator.get(idx)), scale);
    }
}

#define DEFINE_UNARY_KERNEL(func_name, input_type, promote, op)                                                                                                                                                              \
    using func_name##Output = typename promote<input_type>::Output;                                                                                                                                                          \
    extern "C" __global__ void func_name##_contiguous(void *out, void *in, func_name##Output scale, int32_t n)                                                                                                               \
    {                                                                                                                                                                                                                        \
        unary_contiguous<input_type, func_name##Output, op<func_name##Output>>(static_cast<func_name##Output *>(out), static_cast<const input_type *>(in), scale, n, op<func_name##Output>{});                               \
    }                                                                                                                                                                                                                        \
    extern "C" __global__ void func_name##_uncontiguous(void *out, void *in, func_name##Output scale, int32_t size, FastDivmod *in_shape, int32_t *in_strides, int32_t in_ndim)                                              \
    {                                                                                                                                                                                                                        \
        unary_uncontiguous<input_type, func_name##Output, op<func_name##Output>>(static_cast<func_name##Output *>(out), static_cast<input_type *>(in), scale, size, in_shape, in_strides, in_ndim, op<func_name##Output>{}); \
    }

__device__ __forceinline__ f32 celu(f32 a, f32 scale)
{
    f32 gt_mask = (f32)((i32)(a > 0.0));
    return gt_mask * a + (1.0f - gt_mask) * (scale * (expf(a) - 1.0f));
}

__device__ __forceinline__ f64 celu(f64 a, f64 scale)
{
    f64 gt_mask = (f64)((i64)(a > 0.0));
    return gt_mask * a + (1.0 - gt_mask) * (scale * (exp(a) - 1.0));
}

__device__ __forceinline__ f16 celu(f16 a, f16 scale)
{
    f32 res = celu(cast<f16, f32>(a), cast<f16, f32>(scale));
    return cast<f32, f16>(res);
}

__device__ __forceinline__ bf16 celu(bf16 a, bf16 scale)
{
    f32 res = celu(cast<bf16, f32>(a), cast<bf16, f32>(scale));
    return cast<f32, bf16>(res);
}

template <typename Input>
struct Celu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a, Output scale) const
    {
        CHECK_FLOAT_TYPE(Output);
        return celu(cast<Input, Output>(a), scale);
    }
};

DEFINE_UNARY_KERNEL(celu_f16, f16, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_bf16, bf16, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_f32, f32, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_f64, f64, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_bool, bool, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_i8, i8, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_i16, i16, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_i32, i32, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_i64, i64, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_u8, u8, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_u16, u16, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_u32, u32, FloatOutUnaryPromote, Celu);
DEFINE_UNARY_KERNEL(celu_u64, u64, FloatOutUnaryPromote, Celu);
