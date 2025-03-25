#pragma once
#include "../utils/type_cast.cuh"
#include "../utils/fast_divmod.cuh"
#include "../utils/index_calculator.cuh"

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_contiguous(Output *out, const LHS *lhs, const RHS *rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Output>(lhs[idx]), cast<RHS, Output>(rhs[idx]));
    }
}

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_contiguous_lhs_scalar(Output *out, const LHS lhs, const RHS *rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Output lhs_scalar = cast<LHS, Output>(lhs);
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(lhs_scalar, cast<RHS, Output>(rhs[idx]));
    }
}

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_contiguous_rhs_scalar(Output *out, const LHS *lhs, const RHS rhs, int32_t n, Op op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Output rhs_scalar = cast<RHS, Output>(rhs);
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Output>(lhs[idx]), rhs_scalar);
    }
}

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_uncontiguous(
    Output *out,
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
        out[idx] = op(cast<LHS, Output>(lhs_idx_calculator.get(idx)), cast<RHS, Output>(rhs_idx_calculator.get(idx)));
    }
}

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_uncontiguous_lhs_scalar(
    Output *out,
    const LHS lhs,
    RHS *rhs,
    int32_t n,
    Op op,
    FastDivmod *rhs_shape,
    int32_t *rhs_strides,
    int32_t rhs_ndim)
{
    Output lhs_scalar = cast<LHS, Output>(lhs);
    UncontiguousIndexCalculator<RHS> rhs_idx_calculator = UncontiguousIndexCalculator<RHS>(rhs, rhs_shape, rhs_strides, rhs_ndim);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(lhs_scalar, cast<RHS, Output>(rhs_idx_calculator.get(idx)));
    }
}

template <typename LHS, typename RHS, typename Output, typename Op>
__device__ __forceinline__ void binary_uncontiguous_rhs_scalar(
    Output *out,
    LHS *lhs,
    const RHS rhs,
    int32_t n,
    Op op,
    FastDivmod *lhs_shape,
    int32_t *lhs_strides,
    int32_t lhs_ndim)
{
    UncontiguousIndexCalculator<LHS> lhs_idx_calculator = UncontiguousIndexCalculator<LHS>(lhs, lhs_shape, lhs_strides, lhs_ndim);
    Output rhs_scalar = cast<RHS, Output>(rhs);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x)
    {
        out[idx] = op(cast<LHS, Output>(lhs_idx_calculator.get(idx)), rhs_scalar);
    }
}

#define DEFINE_BINARY_KERNEL(func_name, lhs_type, rhs_type, op, promote)                                                                                                                                                                                          \
    extern "C" __global__ void func_name##_contiguous(void *out, void *lhs, void *rhs, int32_t n)                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_contiguous<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), static_cast<const lhs_type *>(lhs), static_cast<const rhs_type *>(rhs), n, op<Output, Output>{});                                                           \
    }                                                                                                                                                                                                                                                             \
    extern "C" __global__ void func_name##_contiguous_lhs_scalar(void *out, lhs_type lhs, void *rhs, int32_t n)                                                                                                                                                   \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_contiguous_lhs_scalar<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), lhs, static_cast<const rhs_type *>(rhs), n, op<Output, Output>{});                                                                               \
    }                                                                                                                                                                                                                                                             \
    extern "C" __global__ void func_name##_contiguous_rhs_scalar(void *out, void *lhs, rhs_type rhs, int32_t n)                                                                                                                                                   \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_contiguous_rhs_scalar<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), static_cast<const lhs_type *>(lhs), rhs, n, op<Output, Output>{});                                                                               \
    }                                                                                                                                                                                                                                                             \
    extern "C" __global__ void func_name##_uncontiguous(void *out, void *lhs, FastDivmod *lhs_shape, int32_t *lhs_strides, void *rhs, FastDivmod *rhs_shape, int32_t *rhs_strides, int32_t lhs_ndim, int32_t rhs_ndim, int32_t n)                                 \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_uncontiguous<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), static_cast<lhs_type *>(lhs), static_cast<rhs_type *>(rhs), n, op<Output, Output>{}, lhs_shape, lhs_strides, rhs_shape, rhs_strides, lhs_ndim, rhs_ndim); \
    }                                                                                                                                                                                                                                                             \
    extern "C" __global__ void func_name##_uncontiguous_lhs_scalar(void *out, lhs_type lhs, void *rhs, FastDivmod *rhs_shape, int32_t *rhs_strides, int32_t rhs_ndim, int32_t n)                                                                                  \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_uncontiguous_lhs_scalar<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), lhs, static_cast<rhs_type *>(rhs), n, op<Output, Output>{}, rhs_shape, rhs_strides, rhs_ndim);                                                 \
    }                                                                                                                                                                                                                                                             \
    extern "C" __global__ void func_name##_uncontiguous_rhs_scalar(void *out, void *lhs, rhs_type rhs, FastDivmod *lhs_shape, int32_t *lhs_strides, int32_t lhs_ndim, int32_t n)                                                                                  \
    {                                                                                                                                                                                                                                                             \
        using Output = typename promote<lhs_type, rhs_type>::Output;                                                                                                                                                                                              \
        binary_uncontiguous_rhs_scalar<lhs_type, rhs_type, Output, op<Output, Output>>(static_cast<Output *>(out), static_cast<lhs_type *>(lhs), rhs, n, op<Output, Output>{}, lhs_shape, lhs_strides, lhs_ndim);                                                 \
    }
