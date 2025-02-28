#pragma once

#include "reduce_helper.cuh"
#include "../utils/index_calculator.cuh"

template <unsigned int WarpSize = 32>
struct Block1D
{
    static __forceinline__ __device__ int Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int Warps()
    {
        return blockDim.x / WarpSize;
    }
};

template <unsigned int WarpSize = 32>
struct Block2D
{
    static __forceinline__ __device__ int Tid()
    {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps()
    {
        return blockDim.x * blockDim.y / WarpSize;
    }
};

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh
template <typename T, typename Op, typename Op2, unsigned int WarpSize = 32, typename B = Block1D<WarpSize>>
__device__ __forceinline__ T blockReduce(T val, T *shared)
{
    const int tid = B::Tid();
    int warp_id;
    int lane_id;
    divmod(tid, (int)WarpSize, warp_id, lane_id);
    val = Op::warp_reduce(val);
    __syncthreads();
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();
    val = (tid < B::Warps()) ? shared[lane_id] : Op::identity();
    if (warp_id == 0)
        val = Op2::warp_reduce(val);
    return val;
}

template <typename R, typename Op, unsigned int WarpSize = 32>
__device__ R block_y_reduce(R value, R *shared)
{
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared[tid] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1)
    {
        __syncthreads();
        if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y)
        {
            R other = shared[tid + offset * blockDim.x];
            value = Op::combine(value, other);
            shared[tid] = value;
        }
    }
    __syncthreads();
    return shared[threadIdx.x];
}

// https://github.com/DefTruth/CUDA-Learn-Notes/blob/main/kernels/reduce/block_all_reduce.cu
template <typename Calculator, typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int BLOCK_SIZE = 256, unsigned int WarpSize = 32>
__device__ void all_reduce(R *out, size_t size, Calculator index_calculator)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    constexpr int NUM_WARPS = (BLOCK_SIZE + WarpSize - 1) / WarpSize;
    __shared__ R reduce_smem[NUM_WARPS];
    while (idx < size)
    {
        R val = Op<T, R, WarpSize>::process_single(index_calculator.get((long long)idx));
        total = Op<R, R, WarpSize>::combine(total, val);
        idx += gridDim.x * blockDim.x;
    }
    int warp = threadIdx.x / WarpSize;
    int lane = threadIdx.x % WarpSize;
    total = Op<R, R, WarpSize>::warp_reduce(total);
    if (lane == 0)
        reduce_smem[warp] = total;
    __syncthreads();
    total = (lane < NUM_WARPS) ? reduce_smem[lane] : Op<R, R, WarpSize>::identity();
    if (warp == 0)
        total = Op<R, R, NUM_WARPS>::warp_reduce(total);
    if (threadIdx.x == 0)
        out[blockIdx.x] = total;
}

template <typename T, typename R, template <typename, typename, uint32_t> class Op = ReduceOp, uint32_t BlockSize, uint32_t WarpSize = 32>
__device__ void reduce_fast_dim_include(R *out, T *in, R *buffer, int32_t *finished, FastDivmod *shape, int32_t *strides, size_t ndim, size_t fast_dim_size, size_t reduce_size_no_fast_dim)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t local_row = threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    auto idx_calculator = FastUncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (local_row < fast_dim_size)
    {
        uint32_t x = blockIdx.x * fast_dim_size + local_row;
        uint32_t idx = x * reduce_size_no_fast_dim;
        const uint32_t stride = blockDim.y * gridDim.y;
        for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < reduce_size_no_fast_dim; i += stride)
        {
            R res = Op<T, R, WarpSize>::process_single(idx_calculator.get(idx + i));
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        local_row += blockDim.x;
    }
    total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block2D<WarpSize>>(total, reduce_smem);
    if (threadIdx.x == 0)
    {
        buffer[blockIdx.x * gridDim.y + blockIdx.y] = total;
    }
    __threadfence();
    __syncthreads();
    if (atomicAdd(&finished[blockIdx.x], 1) == gridDim.y - 1)
    {
        total = Op<R, R, WarpSize>::identity();
        for (uint32_t i = 0; i < gridDim.y; i++)
        {
            total = Op<R, R, WarpSize>::combine(
                total,
                buffer[blockIdx.x * gridDim.y + i]);
        }
        out[blockIdx.x] = total;
    }
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int BlockSize, unsigned int WarpSize = 32>
__device__ void reduce_fast_dim_only(R *out, T *in, R *buffer, int32_t *finished, size_t fast_dim_size, size_t output_size)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t y = blockIdx.y;
    while (y < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t stride = blockDim.x * gridDim.x;
        uint32_t row = y * fast_dim_size;
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < fast_dim_size; i += stride)
        {
            R res = Op<T, R, WarpSize>::process_single(in[row + i]);
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            buffer[y * gridDim.x + blockIdx.x] = total;
        }
        __threadfence();
        __syncthreads();

        if (atomicAdd(&finished[y], 1) == gridDim.x - 1)
        {
            total = Op<R, R, WarpSize>::identity();
            for (uint32_t i = 0; i < gridDim.x; i++)
            {
                total = Op<R, R, WarpSize>::combine(
                    total,
                    buffer[y * gridDim.x + i]);
            }
            out[y] = total;
        }
        y += gridDim.y;
    }
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int BlockSize, unsigned int WarpSize = 32>
__device__ void reduce_small_fast_dim_only(R *out, T *in, size_t fast_dim_size, size_t output_size)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t y = blockIdx.x;
    while (y < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t row = y * fast_dim_size;
        for (uint32_t i = threadIdx.x; i < fast_dim_size; i += blockDim.x)
        {
            R res = Op<T, R, WarpSize>::process_single(in[row + i]);
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
        {
            out[y] = total;
        }
        y += gridDim.x;
    }
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int WarpSize = 32>
__device__ void reduce_fast_dim_not_include(R *out, R *buffer, T *in, int *finished, FastDivmod *shape, int *strides, size_t ndim, size_t reduce_size, size_t output_size)
{
    extern __shared__ R shared[];
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    FastUncontiguousIndexCalculator<T> idx_calculator = FastUncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (x < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t stride = blockDim.y * gridDim.y;
        uint32_t row = x * reduce_size;
        for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < reduce_size; i += stride)
        {
            R res = Op<T, R, WarpSize>::process_single(idx_calculator.get(row + i));
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
            out[x] = total;
        }

        x += blockDim.x * gridDim.x;
    }
}