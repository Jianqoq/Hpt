#pragma once

#include "reduce_helper.cuh"
#include "../utils/index_calculator.cuh"

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

// size include no reduce dim and fast dim
template <typename T, typename R, typename ProgressUpdater, template <typename, typename, unsigned int> class Op = ReduceOp, unsigned int WarpSize = 32>
__device__ void reduce_fast_dim_include(R *out, T *in, long long *shape, long long *strides, size_t ndim, size_t fast_dim_size, size_t num_elements_per_thread, ProgressUpdater progress_updater)
{
    unsigned int thread_offset = threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    while (thread_offset < fast_dim_size)
    {
        unsigned int idx = (blockIdx.x * fast_dim_size + thread_offset) * num_elements_per_thread;
        T *in_ptr = UncontiguousIndexCalculator<T>(in, shape, strides, (int)ndim).get_ptr(idx);
        progress_updater.set_ptr(in_ptr);
        R sub_total = Op<R, R, WarpSize>::identity();
        for (size_t i = 0; i < num_elements_per_thread; i++)
        {
            T val = progress_updater.get();
            R res = Op<T, R, WarpSize>::process_single(val);
            progress_updater.update();
            sub_total = Op<R, R, WarpSize>::combine(sub_total, res);
        }
        total = Op<R, R, WarpSize>::combine(total, sub_total);
        thread_offset += blockDim.x;
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = total;
}