#pragma once

#include "reduce_helper.cuh"
#include "../utils/index_calculator.cuh"
#include "block_reduce.cuh"

// https://github.com/DefTruth/CUDA-Learn-Notes/blob/main/kernels/reduce/block_all_reduce.cu
template <typename Calculator, typename T, typename R, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t BLOCK_SIZE = 256, uint32_t WarpSize = 32>
__device__ void all_reduce(R *out, R *buffer, int32_t *finished, size_t size, Calculator index_calculator)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    constexpr int32_t NUM_WARPS = (BLOCK_SIZE + WarpSize - 1) / WarpSize;
    __shared__ R reduce_smem[NUM_WARPS];
    while (idx < size)
    {
        R val = Op<T, R, WarpSize>::pre_op(index_calculator.get((long long)idx));
        total = Op<R, R, WarpSize>::combine(total, val);
        idx += gridDim.x * blockDim.x;
    }
    int32_t warp = threadIdx.x / WarpSize;
    int32_t lane = threadIdx.x % WarpSize;
    total = Op<R, R, WarpSize>::warp_reduce(total);
    if (lane == 0)
        reduce_smem[warp] = total;
    __syncthreads();
    total = (lane < NUM_WARPS) ? reduce_smem[lane] : Op<R, R, WarpSize>::identity();
    if (warp == 0)
        total = Op<R, R, NUM_WARPS>::warp_reduce(total);
    if (threadIdx.x == 0)
        buffer[blockIdx.x] = total;
    __threadfence();

    bool is_last = is_last_block(&finished[0], gridDim.x);

    if (is_last)
    {
        __threadfence();
        total = Op<R, R, WarpSize>::identity();
        for (uint32_t i = threadIdx.x; i < gridDim.x; i += blockDim.x)
        {
            total = Op<R, R, WarpSize>::combine(total, buffer[i]);
        }
        __syncthreads();
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BLOCK_SIZE / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
            out[0] = Op<R, R, WarpSize>::post_op(total, size);
    }
}

template <typename T, typename R, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_include(R *out, T *in, R *buffer, int32_t *finished, FastDivmod *shape, int32_t *strides, size_t ndim, size_t fast_dim_size, size_t reduce_size_no_fast_dim)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t local_row = threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    auto idx_calculator = UncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (local_row < fast_dim_size)
    {
        uint32_t x = blockIdx.x * fast_dim_size + local_row;
        uint32_t idx = x * reduce_size_no_fast_dim;
        const uint32_t stride = blockDim.y * gridDim.y;
        for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < reduce_size_no_fast_dim; i += stride)
        {
            R res = Op<T, R, WarpSize>::pre_op(idx_calculator.get(idx + i));
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        local_row += blockDim.x;
    }
    total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(total, reduce_smem);
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        buffer[blockIdx.x * gridDim.y + blockIdx.y] = total;
    }
    total = Op<R, R, WarpSize>::identity();
    __threadfence();
    bool is_last = is_last_block(&finished[blockIdx.x], gridDim.y);
    if (is_last)
    {
        __threadfence();
        if (threadIdx.x == 0)
        {
            for (uint32_t i = threadIdx.y; i < gridDim.y; i += blockDim.y)
            {
                total = Op<R, R, WarpSize>::combine(
                    total,
                    buffer[blockIdx.x * gridDim.y + i]);
            }
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            out[blockIdx.x] = Op<R, R, WarpSize>::post_op(total, fast_dim_size * reduce_size_no_fast_dim);
        }
    }
}

template <typename T, typename R, typename CalIndex, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_only(R *out, T *in, R *buffer, int32_t *finished, size_t fast_dim_size, size_t output_size, CalIndex index_calculator, int64_t last_stride)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t y = blockIdx.y;
    while (y < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t stride = blockDim.x * gridDim.x;
        uint32_t row = y * fast_dim_size;
        in = index_calculator.get_ptr(row);
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < fast_dim_size; i += stride)
        {
            R res = Op<T, R, WarpSize>::pre_op(in[i * last_stride]);
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            buffer[y * gridDim.x + blockIdx.x] = total;
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0 && atomicAdd(&finished[y], 1) == gridDim.x - 1)
        {
            total = Op<R, R, WarpSize>::identity();
            for (uint32_t i = 0; i < gridDim.x; i++)
            {
                total = Op<R, R, WarpSize>::combine(
                    total,
                    buffer[y * gridDim.x + i]);
            }
            out[y] = Op<R, R, WarpSize>::post_op(total, fast_dim_size);
        }
        y += gridDim.y;
    }
}

template <typename T, typename R, typename CalIndex, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_small_fast_dim_only(R *out, T *in, size_t fast_dim_size, size_t output_size, CalIndex index_calculator, int64_t last_stride)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t y = blockIdx.x;
    while (y < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t row = y * fast_dim_size;
        in = index_calculator.get_ptr(row);
        for (uint32_t i = threadIdx.x; i < fast_dim_size; i += blockDim.x)
        {
            R res = Op<T, R, WarpSize>::pre_op(in[i * last_stride]);
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            out[y] = Op<R, R, WarpSize>::post_op(total, fast_dim_size);
        }
        y += gridDim.x;
    }
}

template <typename T, typename R, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_not_include(R *shared, R *out, R *buffer, T *in, int32_t *finished, FastDivmod *shape, int32_t *strides, size_t ndim, size_t reduce_size, size_t output_size)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    UncontiguousIndexCalculator<T> idx_calculator = UncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (x < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t stride = blockDim.y * gridDim.y;
        uint32_t row = x * reduce_size;
        for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < reduce_size; i += stride)
        {
            R res = Op<T, R, WarpSize>::pre_op(idx_calculator.get(row + i));
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = block_y_reduce<R, Op<R, R, WarpSize>, WarpSize>(total, shared);
        if (threadIdx.y == 0)
        {
            buffer[x * gridDim.y + blockIdx.y] = total;
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.y == 0 && atomicAdd(&finished[x], 1) == gridDim.y - 1)
        {
            total = Op<R, R, WarpSize>::identity();
            for (uint32_t i = 0; i < gridDim.y; i++)
            {
                total = Op<R, R, WarpSize>::combine(
                    total,
                    buffer[x * gridDim.y + i]);
            }
            out[x] = Op<R, R, WarpSize>::post_op(total, reduce_size);
        }

        x += blockDim.x * gridDim.x;
    }
}