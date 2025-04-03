#pragma once
#include "../utils/fast_divmod.cuh"

template <uint32_t WarpSize = 32>
struct Block1D
{
    static __forceinline__ __device__ int32_t Tid() { return threadIdx.x; }

    static __forceinline__ __device__ int32_t Warps()
    {
        return blockDim.x / WarpSize;
    }
};

template <uint32_t WarpSize = 32>
struct Block2D
{
    static __forceinline__ __device__ int32_t Tid()
    {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int32_t Warps()
    {
        return blockDim.x * blockDim.y / WarpSize;
    }
};

template <typename R, typename Op, uint32_t WarpSize = 32>
__device__ R block_y_reduce(R value, R *shared)
{
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared[tid] = value;
    for (int32_t offset = blockDim.y / 2; offset > 0; offset >>= 1)
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

// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh
template <typename T, typename Op, typename Op2, uint32_t WarpSize = 32, typename B = Block1D<WarpSize>>
__device__ __forceinline__ T blockReduce(T val, T *shared)
{
    const int32_t tid = B::Tid();
    int32_t warp_id;
    int32_t lane_id;
    divmod(tid, (int32_t)WarpSize, warp_id, lane_id);
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

template <typename T, typename Op, typename Op2, uint32_t WarpSize = 32, typename B = Block1D<WarpSize>>
__device__ __forceinline__ T ArgBlockReduce(T val, T *shared)
{
    const int32_t tid = B::Tid();
    int32_t warp_id;
    int32_t lane_id;
    divmod(tid, (int32_t)WarpSize, warp_id, lane_id);
    val = Op::warp_reduce(val);
    __syncthreads();
    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();
    val = (tid < B::Warps()) ? shared[lane_id] : T::identity();
    if (warp_id == 0)
        val = Op2::warp_reduce(val);
    return val;
}