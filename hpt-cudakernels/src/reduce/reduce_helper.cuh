#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T, typename R, unsigned int WarpSize>
class ReduceOp
{
public:
    __device__ __forceinline__ static R combine(T a, R b) { return R(); };
    __device__ __forceinline__ static T identity() { return T(); };
    __device__ __forceinline__ static R warp_reduce(R a) { return R(); }
    __device__ __forceinline__ static R pre_op(T a) { return a; }
    __device__ __forceinline__ static R post_op(R a, size_t reduce_size) { return a; }
};

__device__ __forceinline__ bool is_last_block(int32_t *finished, size_t size)
{
    __shared__ bool is_last;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int32_t tmp = atomicAdd(finished, 1);
        is_last = tmp == (size - 1);
    }
    __syncthreads();
    return is_last;
}
