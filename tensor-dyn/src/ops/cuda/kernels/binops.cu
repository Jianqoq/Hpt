#include "utils.cuh"

extern "C" __global__ void lhs_scalar_rhs_contiguous(float *out, float lhs, float *rhs, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < size)
    {
        out[idx] = lhs * rhs[idx];
        idx += stride;
    }
}

extern "C" __global__ void lhs_scalar_rhs_not_contiguous(float *out, float lhs, float *rhs, size_t size, size_t ndim, long long *shape, long long *strides)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const long long shape[] = {0, 1, 2, };
    while (idx < size)
    {
        long amount = idx;
        long offset = 0;
        for (int j = ndim - 1; j >= 0; j--)
        {
            offset += amount % shape[j] * strides[j];
            amount /= shape[j];
        }
        out[idx] = lhs * rhs[offset];
        idx += stride;
    }
}

extern "C" __global__ void rhs_scalar(float *out, float *lhs, float rhs, size_t size, size_t ndim, size_t *shape, size_t *strides)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    if (is_contiguous(ndim, strides))
    {
        while (idx < size)
        {
            out[idx] = lhs[idx] * rhs;
            idx += stride;
        }
    }
    else
    {
        while (idx < size)
        {
            long amount = idx;
            long offset = 0;
            for (int j = ndim - 1; j >= 0; j--)
            {
                offset += amount % shape[j] * strides[j];
                amount /= shape[j];
            }
            out[idx] = lhs[offset] * rhs;
            idx += stride;
        }
    }
}

extern "C" __global__ void binary_normal_contiguous(float *out, float *lhs, float *rhs, size_t size, size_t ndim, size_t *shape)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < size)
    {
        out[idx] = lhs[idx] * rhs[idx];
        idx += stride;
    }
}

extern "C" __global__ void binary_normal_non_contiguous(float *out, float *lhs, float *rhs, size_t size, size_t ndim, size_t *shape, size_t *lhs_strides, size_t *rhs_strides)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    while (idx < size)
    {
        long lhs_amount = idx;
        long rhs_amount = idx;
        long lhs_offset = 0;
        long rhs_offset = 0;
        for (int j = ndim - 1; j >= 0; j--)
        {
            lhs_offset += lhs_amount % shape[j] * lhs_strides[j];
            rhs_offset += rhs_amount % shape[j] * rhs_strides[j];
            lhs_amount /= shape[j];
            rhs_amount /= shape[j];
        }
        out[idx] = lhs[lhs_offset] * rhs[rhs_offset];
        idx += stride;
    }
}