#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_contiguous(bool *out, const LHS *lhs, const RHS *rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Intermediate>(lhs[idx]), cast<RHS, Intermediate>(rhs[idx]));
    }
}

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_contiguous_lhs_scalar(bool *out, const LHS lhs, const RHS *rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Intermediate lhs_scalar = cast<LHS, Intermediate>(lhs);
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(lhs_scalar, cast<RHS, Intermediate>(rhs[idx]));
    }
}

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_contiguous_rhs_scalar(bool *out, const LHS *lhs, const RHS rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Intermediate rhs_scalar = cast<RHS, Intermediate>(rhs);
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Intermediate>(lhs[idx]), rhs_scalar);
    }
}

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_uncontiguous(
    bool *out,
    LHS *lhs,
    RHS *rhs,
    int32_t n,
    Op op,
    FastDivmod *lhs_shape,
    int32_t *lhs_strides,
    FastDivmod *rhs_shape,
    int32_t *rhs_strides,
    int32_t lhs_ndim,
    int32_t rhs_ndim)
{
    UncontiguousIndexCalculator<LHS> lhs_idx_calculator = UncontiguousIndexCalculator<LHS>(lhs, lhs_shape, lhs_strides, lhs_ndim);
    UncontiguousIndexCalculator<RHS> rhs_idx_calculator = UncontiguousIndexCalculator<RHS>(rhs, rhs_shape, rhs_strides, rhs_ndim);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Intermediate>(lhs_idx_calculator.get(idx)), cast<RHS, Intermediate>(rhs_idx_calculator.get(idx)));
    }
}

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_uncontiguous_lhs_scalar(
    bool *out,
    const LHS lhs,
    RHS *rhs,
    int32_t n,
    Op op,
    FastDivmod *rhs_shape,
    int32_t *rhs_strides,
    int32_t rhs_ndim)
{
    Intermediate lhs_scalar = cast<LHS, Intermediate>(lhs);
    UncontiguousIndexCalculator<RHS> rhs_idx_calculator = UncontiguousIndexCalculator<RHS>(rhs, rhs_shape, rhs_strides, rhs_ndim);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(lhs_scalar, cast<RHS, Intermediate>(rhs_idx_calculator.get(idx)));
    }
}

template <typename LHS, typename RHS, typename Intermediate, typename Op>
__device__ __forceinline__ void cmp_uncontiguous_rhs_scalar(
    bool *out,
    LHS *lhs,
    const RHS rhs,
    int32_t n,
    Op op,
    FastDivmod *lhs_shape,
    int32_t *lhs_strides,
    int32_t lhs_ndim)
{
    UncontiguousIndexCalculator<LHS> lhs_idx_calculator = UncontiguousIndexCalculator<LHS>(lhs, lhs_shape, lhs_strides, lhs_ndim);
    Intermediate rhs_scalar = cast<RHS, Intermediate>(rhs);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Intermediate>(lhs_idx_calculator.get(idx)), rhs_scalar);
    }
}

#define DEFINE_CMP_KERNEL(func_name, lhs_type, rhs_type, op, promote)                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_contiguous(bool *out, void *lhs, void *rhs, int32_t n)                                                                                                                                                                     \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_contiguous<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, static_cast<const lhs_type *>(lhs), static_cast<const rhs_type *>(rhs), n, op<Intermediate, Intermediate>{});                                                           \
    }                                                                                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_contiguous_lhs_scalar(bool *out, lhs_type lhs, void *rhs, int32_t n)                                                                                                                                                       \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_contiguous_lhs_scalar<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, lhs, static_cast<const rhs_type *>(rhs), n, op<Intermediate, Intermediate>{});                                                                               \
    }                                                                                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_contiguous_rhs_scalar(bool *out, void *lhs, rhs_type rhs, int32_t n)                                                                                                                                                       \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_contiguous_rhs_scalar<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, static_cast<const lhs_type *>(lhs), rhs, n, op<Intermediate, Intermediate>{});                                                                               \
    }                                                                                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_uncontiguous(bool *out, void *lhs, FastDivmod *lhs_shape, int32_t *lhs_strides, void *rhs, FastDivmod *rhs_shape, int32_t *rhs_strides, int32_t lhs_ndim, int32_t rhs_ndim, int32_t n)                                     \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_uncontiguous<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, static_cast<lhs_type *>(lhs), static_cast<rhs_type *>(rhs), n, op<Intermediate, Intermediate>{}, lhs_shape, lhs_strides, rhs_shape, rhs_strides, lhs_ndim, rhs_ndim); \
    }                                                                                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_uncontiguous_lhs_scalar(bool *out, lhs_type lhs, void *rhs, FastDivmod *rhs_shape, int32_t *rhs_strides, int32_t rhs_ndim, int32_t n)                                                                                      \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_uncontiguous_lhs_scalar<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, lhs, static_cast<rhs_type *>(rhs), n, op<Intermediate, Intermediate>{}, rhs_shape, rhs_strides, rhs_ndim);                                                 \
    }                                                                                                                                                                                                                                                                 \
    extern "C" __global__ void func_name##_uncontiguous_rhs_scalar(bool *out, void *lhs, rhs_type rhs, FastDivmod *lhs_shape, int32_t *lhs_strides, int32_t lhs_ndim, int32_t n)                                                                                      \
    {                                                                                                                                                                                                                                                                 \
        using Intermediate = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                            \
        cmp_uncontiguous_rhs_scalar<lhs_type, rhs_type, Intermediate, op<Intermediate, Intermediate>>(out, static_cast<lhs_type *>(lhs), rhs, n, op<Intermediate, Intermediate>{}, lhs_shape, lhs_strides, lhs_ndim);                                                 \
    }

template <typename LHS, typename RHS>
struct Eq
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) == cast<RHS, Output>(b);
    }
};

template <typename LHS, typename RHS>
struct Ne
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) != cast<RHS, Output>(b);
    }
};

template <typename LHS, typename RHS>
struct Lt
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) < cast<RHS, Output>(b);
    }
};

template <typename LHS, typename RHS>
struct Le
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) <= cast<RHS, Output>(b);
    }
};

template <typename LHS, typename RHS>
struct Gt
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) > cast<RHS, Output>(b);
    }
};

template <typename LHS, typename RHS>
struct Ge
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        return cast<LHS, Output>(a) >= cast<RHS, Output>(b);
    }
};

#define DEFINE_KERNELS(func_name, op)                                           \
    DEFINE_CMP_KERNEL(func_name##_bool_bool, bool, bool, op, NormalOutPromote); \
    DEFINE_CMP_KERNEL(func_name##_bool_i8, bool, i8, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_bool_i16, bool, i16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_i32, bool, i32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_i64, bool, i64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_u8, bool, u8, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_bool_u16, bool, u16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_u32, bool, u32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_u64, bool, u64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_f16, bool, f16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_bf16, bool, bf16, op, NormalOutPromote); \
    DEFINE_CMP_KERNEL(func_name##_bool_f32, bool, f32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bool_f64, bool, f64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i8_bool, i8, bool, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i8_i8, i8, i8, op, NormalOutPromote);         \
    DEFINE_CMP_KERNEL(func_name##_i8_i16, i8, i16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_i32, i8, i32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_i64, i8, i64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_u8, i8, u8, op, NormalOutPromote);         \
    DEFINE_CMP_KERNEL(func_name##_i8_u16, i8, u16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_u32, i8, u32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_u64, i8, u64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_f16, i8, f16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_bf16, i8, bf16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i8_f32, i8, f32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i8_f64, i8, f64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i16_bool, i16, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i16_i8, i16, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i16_i16, i16, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_i32, i16, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_i64, i16, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_u8, i16, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i16_u16, i16, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_u32, i16, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_u64, i16, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_f16, i16, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_bf16, i16, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i16_f32, i16, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i16_f64, i16, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_bool, i32, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i32_i8, i32, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i32_i16, i32, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_i32, i32, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_i64, i32, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_u8, i32, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i32_u16, i32, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_u32, i32, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_u64, i32, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_f16, i32, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_bf16, i32, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i32_f32, i32, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i32_f64, i32, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_bool, i64, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i64_i8, i64, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i64_i16, i64, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_i32, i64, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_i64, i64, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_u8, i64, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_i64_u16, i64, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_u32, i64, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_u64, i64, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_f16, i64, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_bf16, i64, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_i64_f32, i64, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_i64_f64, i64, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u8_bool, u8, bool, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u8_i8, u8, i8, op, NormalOutPromote);         \
    DEFINE_CMP_KERNEL(func_name##_u8_i16, u8, i16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_i32, u8, i32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_i64, u8, i64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_u8, u8, u8, op, NormalOutPromote);         \
    DEFINE_CMP_KERNEL(func_name##_u8_u16, u8, u16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_u32, u8, u32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_u64, u8, u64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_f16, u8, f16, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_bf16, u8, bf16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u8_f32, u8, f32, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u8_f64, u8, f64, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u16_bool, u16, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u16_i8, u16, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u16_i16, u16, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_i32, u16, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_i64, u16, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_u8, u16, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u16_u16, u16, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_u32, u16, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_u64, u16, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_f16, u16, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_bf16, u16, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u16_f32, u16, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u16_f64, u16, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_bool, u32, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u32_i8, u32, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u32_i16, u32, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_i32, u32, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_i64, u32, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_u8, u32, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u32_u16, u32, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_u32, u32, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_u64, u32, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_f16, u32, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_bf16, u32, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u32_f32, u32, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u32_f64, u32, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_bool, u64, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u64_i8, u64, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u64_i16, u64, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_i32, u64, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_i64, u64, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_u8, u64, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_u64_u16, u64, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_u32, u64, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_u64, u64, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_f16, u64, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_bf16, u64, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_u64_f32, u64, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_u64_f64, u64, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_bool, f16, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f16_i8, f16, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f16_i16, f16, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_i32, f16, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_i64, f16, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_u8, f16, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f16_u16, f16, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_u32, f16, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_u64, f16, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_f16, f16, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_bf16, f16, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f16_f32, f16, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f16_f64, f16, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_bf16_bool, bf16, bool, op, NormalOutPromote); \
    DEFINE_CMP_KERNEL(func_name##_bf16_i8, bf16, i8, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_bf16_i16, bf16, i16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_i32, bf16, i32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_i64, bf16, i64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_u8, bf16, u8, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_bf16_u16, bf16, u16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_u32, bf16, u32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_u64, bf16, u64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_f16, bf16, f16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_f32, bf16, f32, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_bf16_f64, bf16, f64, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f32_bool, f32, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f32_i8, f32, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f32_i16, f32, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_i32, f32, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_i64, f32, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_u8, f32, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f32_u16, f32, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_u32, f32, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_u64, f32, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_f16, f32, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_bf16, f32, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f32_f32, f32, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f32_f64, f32, f64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_bool, f64, bool, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f64_i8, f64, i8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f64_i16, f64, i16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_i32, f64, i32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_i64, f64, i64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_u8, f64, u8, op, NormalOutPromote);       \
    DEFINE_CMP_KERNEL(func_name##_f64_u16, f64, u16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_u32, f64, u32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_u64, f64, u64, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_f16, f64, f16, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_bf16, f64, bf16, op, NormalOutPromote);   \
    DEFINE_CMP_KERNEL(func_name##_f64_f32, f64, f32, op, NormalOutPromote);     \
    DEFINE_CMP_KERNEL(func_name##_f64_f64, f64, f64, op, NormalOutPromote);

DEFINE_KERNELS(eq, Eq);
DEFINE_KERNELS(ne, Ne);
DEFINE_KERNELS(lt, Lt);
DEFINE_KERNELS(le, Le);
DEFINE_KERNELS(gt, Gt);
DEFINE_KERNELS(ge, Ge);
