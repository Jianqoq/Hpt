#pragma once

#include "reduce_helper.cuh"
#include "../utils/index_calculator.cuh"

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh
template <typename T, typename Op, typename Op2, unsigned int BLOCK_SIZE = 256, unsigned int WarpSize = 32>
__device__ __forceinline__ T blockReduce(T val, T *shared)
{
    int tid = threadIdx.x;
    int warp_id;
    int lane_id;
    divmod(tid, (int)WarpSize, warp_id, lane_id);
    val = Op::warp_reduce(val);
    __syncthreads();
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();
    val = (tid < BLOCK_SIZE / WarpSize) ? shared[lane_id] : Op::identity();
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

// size include no reduce dim and fast dim
template <typename T, typename R, template <typename, typename, unsigned int> class Op = ReduceOp, typename ProgressUpdater, unsigned int BLOCK_SIZE = 256, unsigned int WarpSize = 32, bool fast_path = false>
__device__ void reduce_fast_dim_include(R *out, T *in, long long *shape, long long *strides, int ndim, size_t fast_dim_size, size_t num_elements_per_thread, ProgressUpdater progress_updater)
{
    __shared__ R reduce_smem[WarpSize];
    unsigned int thread_offset = threadIdx.x;
    R total = Op<R, R, WarpSize>::identity();
    auto idx_calculator = UncontiguousIndexCalculator<T>(in, shape, strides, ndim);
    while (thread_offset < fast_dim_size)
    {
        unsigned int idx = (blockIdx.x * fast_dim_size + thread_offset) * num_elements_per_thread;
        in = idx_calculator.get_ptr(idx);
        progress_updater.set_ptr(in);
        for (size_t i = 0; i < num_elements_per_thread; i++)
        {
            R res = Op<T, R, WarpSize>::process_single(progress_updater.get());
            total = Op<R, R, WarpSize>::combine(total, res);
            if (fast_path)
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
    total = blockReduce<R, Op<R, R, WarpSize>, Op<R, R, BLOCK_SIZE / WarpSize>, BLOCK_SIZE, WarpSize>(total, reduce_smem);
    if (threadIdx.x == 0)
        out[blockIdx.x] = total;
}