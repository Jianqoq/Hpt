#pragma once

#include "reduce_helper.cuh"
#include "../utils/index_calculator.cuh"
#include "block_reduce.cuh"

template <typename Calculator, typename T, typename AR, template <typename, uint32_t> class Op, uint32_t BLOCK_SIZE = 256, uint32_t WarpSize = 32>
__device__ void all_reduce(int64_t *out, AR *buffer, int32_t *finished, size_t size, Calculator index_calculator)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    AR total = AR::identity();
    constexpr int32_t NUM_WARPS = (BLOCK_SIZE + WarpSize - 1) / WarpSize;
    __shared__ AR reduce_smem[NUM_WARPS];
    while (idx < size)
    {
        T val = index_calculator.get((int64_t)idx);
        total = Op<T, WarpSize>::combine(total, AR(val, (int64_t)idx));
        idx += gridDim.x * blockDim.x;
    }
    int32_t warp = threadIdx.x / WarpSize;
    int32_t lane = threadIdx.x % WarpSize;
    total = Op<T, WarpSize>::warp_reduce(total);
    if (lane == 0)
        reduce_smem[warp] = total;
    __syncthreads();
    total = (lane < NUM_WARPS) ? reduce_smem[lane] : AR::identity();
    if (warp == 0)
        total = Op<T, NUM_WARPS>::warp_reduce(total);
    if (threadIdx.x == 0)
        buffer[blockIdx.x] = total;
    __threadfence();

    bool is_last = is_last_block(&finished[0], gridDim.x);

    if (is_last)
    {
        __threadfence();
        total = AR::identity();
        for (uint32_t i = threadIdx.x; i < gridDim.x; i += blockDim.x)
        {
            total = Op<T, WarpSize>::combine(total, buffer[i]);
        }
        __syncthreads();
        total = ArgBlockReduce<AR, Op<T, WarpSize>, Op<T, BLOCK_SIZE / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
            out[0] = total.idx;
    }
}

template <typename T, typename AR, typename CalIndex, template <typename, uint32_t> class Op, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_only(int64_t *out, T *in, AR *buffer, int32_t *finished, size_t fast_dim_size, size_t output_size, CalIndex index_calculator, int64_t last_stride)
{
    __shared__ AR reduce_smem[WarpSize];
    uint32_t y = blockIdx.y;
    while (y < output_size)
    {
        AR total = AR::identity();
        uint32_t stride = blockDim.x * gridDim.x;
        uint32_t row = y * fast_dim_size;
        in = index_calculator.get_ptr(row);
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < fast_dim_size; i += stride)
        {
            T res = in[i * last_stride];
            total = Op<T, WarpSize>::combine(total, AR(res, (int64_t)(i)));
        }
        total = ArgBlockReduce<AR, Op<T, WarpSize>, Op<T, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            buffer[y * gridDim.x + blockIdx.x] = total;
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0 && atomicAdd(&finished[y], 1) == gridDim.x - 1)
        {
            total = AR::identity();
            for (uint32_t i = 0; i < gridDim.x; i++)
            {
                total = Op<T, WarpSize>::combine(
                    total,
                    buffer[y * gridDim.x + i]);
            }
            out[y] = total.idx;
        }
        y += gridDim.y;
    }
}

template <typename T, typename AR, typename CalIndex, template <typename, uint32_t> class Op, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_small_fast_dim_only(int64_t *out, T *in, size_t fast_dim_size, size_t output_size, CalIndex index_calculator, int64_t last_stride)
{
    __shared__ AR reduce_smem[WarpSize];
    uint32_t y = blockIdx.x;
    while (y < output_size)
    {
        AR total = AR::identity();
        uint32_t row = y * fast_dim_size;
        in = index_calculator.get_ptr(row);
        for (uint32_t i = threadIdx.x; i < fast_dim_size; i += blockDim.x)
        {
            T res = in[i * last_stride];
            total = Op<T, WarpSize>::combine(total, AR(res, (int64_t)(i)));
        }
        total = ArgBlockReduce<AR, Op<T, WarpSize>, Op<T, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            out[y] = total.idx;
        }
        y += gridDim.x;
    }
}

template <typename T, typename AR, template <typename, uint32_t> class Op, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_not_include(AR *shared, int64_t *out, AR *buffer, T *in, int32_t *finished, FastDivmod *shape, int32_t *strides, size_t ndim, size_t reduce_size, size_t output_size)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    UncontiguousIndexCalculator<T> idx_calculator = UncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (x < output_size)
    {
        AR total = AR::identity();
        uint32_t stride = blockDim.y * gridDim.y;
        uint32_t row = x * reduce_size;
        for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < reduce_size; i += stride)
        {
            T res = idx_calculator.get(row + i);
            total = Op<T, WarpSize>::combine(total, AR(res, (int64_t)(i)));
        }
        total = block_y_reduce<AR, Op<T, WarpSize>, WarpSize>(total, shared);
        if (threadIdx.y == 0)
        {
            buffer[x * gridDim.y + blockIdx.y] = total;
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.y == 0 && atomicAdd(&finished[x], 1) == gridDim.y - 1)
        {
            total = AR::identity();
            for (uint32_t i = 0; i < gridDim.y; i++)
            {
                total = Op<T, WarpSize>::combine(
                    total,
                    buffer[x * gridDim.y + i]);
            }
            out[x] = total.idx;
        }

        x += blockDim.x * gridDim.x;
    }
}