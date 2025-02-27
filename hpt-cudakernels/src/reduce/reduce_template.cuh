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

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, typename ProgressUpdater, unsigned int WarpSize = 32, bool fast_path = false>
__device__ void reduce_fast_dim_include(R *out, T *in, FastDivmod *shape, int *strides, int ndim, size_t fast_dim_size, size_t num_elements_per_thread, size_t reduce_size_no_fast_dim, ProgressUpdater progress_updater)
{
    __shared__ R reduce_smem[WarpSize];
    unsigned int thread_offset = threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    auto idx_calculator = FastUncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (thread_offset < fast_dim_size)
    {
        unsigned int horizontal_offset = threadIdx.y * num_elements_per_thread;
        unsigned int idx = (blockIdx.x * fast_dim_size + thread_offset) * reduce_size_no_fast_dim;
        in = idx_calculator.get_ptr(idx + horizontal_offset);
        progress_updater.set_ptr(in);
        for (unsigned int i = horizontal_offset; i < horizontal_offset + num_elements_per_thread && i < reduce_size_no_fast_dim; i++)
        {
            R res = Op<T, R, WarpSize>::process_single(progress_updater.get());
            total = Op<R, R, WarpSize>::combine(total, res);
            if constexpr (fast_path)
            {
                in += strides[ndim - 1];
                progress_updater.set_ptr(in);
            }
            else
            {
                progress_updater.update();
            }
        }
        thread_offset += blockDim.x;
    }
    total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, 512 / WarpSize>, WarpSize, Block2D<WarpSize>>(total, reduce_smem);
    if (threadIdx.x == 0)
        out[blockIdx.x] = total;
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int BlockSize = 256, unsigned int WarpSize = 32>
__device__ void reduce_fast_dim_only(R *out, T *in, size_t fast_dim_size, size_t output_size)
{
    __shared__ R reduce_smem[WarpSize];
    unsigned int block_idx = blockIdx.x;
    while (block_idx < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        unsigned int thread_offset = threadIdx.x;
        while (thread_offset < fast_dim_size)
        {
            unsigned int idx = block_idx * fast_dim_size + thread_offset;
            R res = Op<T, R, WarpSize>::process_single(in[idx]);
            total = Op<R, R, WarpSize>::combine(total, res);
            thread_offset += blockDim.x;
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BlockSize / WarpSize>, WarpSize, Block1D<WarpSize>>(total, reduce_smem);
        if (threadIdx.x == 0)
            out[block_idx] = total;
        block_idx += gridDim.x;
    }
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, typename ProgressUpdater, unsigned int WarpSize = 32, bool fast_path = false>
__device__ void reduce_fast_dim_not_include(R *out, T *in, FastDivmod *shape, int *strides, int ndim, size_t reduce_size, ProgressUpdater progress_updater, size_t output_size)
{
    R total = Op<R, R, WarpSize>::identity();
    auto idx_calculator = FastUncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * reduce_size;
    if (blockIdx.x * blockDim.x + threadIdx.x >= output_size)
        return;
    in = idx_calculator.get_ptr(idx);
    progress_updater.set_ptr(in);
    for (unsigned int i = 0; i < reduce_size; i++)
    {
        R res = Op<T, R, WarpSize>::process_single(progress_updater.get());
        total = Op<R, R, WarpSize>::combine(total, res);
        if constexpr (fast_path)
        {
            in += strides[ndim - 1];
            progress_updater.set_ptr(in);
        }
        else
        {
            progress_updater.update();
        }
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = total;
}

template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int WarpSize = 32>
__device__ void reduce_fast_dim_not_include_small(
    R *out,
    T *in,
    FastDivmod *shape,
    int *strides,
    int ndim,
    size_t reduce_size,
    size_t output_size,
    size_t num_el_per_thread)
{
    __shared__ R reduce_smem[WarpSize];
    uint32_t x = blockIdx.x;
    FastUncontiguousIndexCalculator<T> idx_calculator = FastUncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (x < output_size)
    {
        R total = Op<R, R, WarpSize>::identity();
        uint32_t y = threadIdx.x;
        uint32_t global_idx = x * reduce_size;
        y *= num_el_per_thread;
        for (uint32_t i = y; i < y + num_el_per_thread && i < reduce_size; i++)
        {
            uint32_t idx = global_idx + i;
            T val = idx_calculator.get(idx);
            R res = Op<T, R, WarpSize>::process_single(val);
            total = Op<R, R, WarpSize>::combine(total, res);
        }
        total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, 1024 / WarpSize>, WarpSize>(total, reduce_smem);
        out[x] = total;
        x += gridDim.x * blockDim.x;
    }
}