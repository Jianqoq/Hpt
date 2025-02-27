#pragma once

#include <cuda_fp16.h>

template <typename T, typename R, unsigned int WarpSize>
class ReduceOp
{
public:
    __device__ __forceinline__ static R combine(T a, R b) { return R(); };
    __device__ __forceinline__ static T identity() { return T(); };
    __device__ __forceinline__ static R warp_reduce(R a) { return R(); }
    __device__ __forceinline__ static R warp_reduce_16(R a) { return R(); }
    __device__ __forceinline__ static R process_single(T a) { return R(); }
};