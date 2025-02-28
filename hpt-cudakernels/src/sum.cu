#include "reduce/reduce_template.cuh"
#include "reduce/reduce_helper.cuh"
#include "utils/loop_progress.cuh"
#include <stdint.h>
#include "utils/fast_divmod.cuh"
#include <stdio.h>

template <typename T, typename R = T, unsigned int WarpSize = 32>
class Sum : public ReduceOp<T, R, WarpSize>
{
public:
    __device__ __forceinline__ static T combine(T a, T b)
    {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hadd(a, b);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a || b;
        }
        else
        {
            return a + b;
        }
    }

    __device__ __forceinline__ static T identity()
    {
        return T(0);
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a += __shfl_xor_sync(0xffffffff, a, mask);
        }
        return a;
    }

    __device__ __forceinline__ static R warp_reduce_16(R a)
    {
        a += __shfl_xor_sync(0xffffffff, a, 16);
        return a;
    }

    __device__ __forceinline__ static R process_single(T a)
    {
        return a;
    }
};

extern "C" __global__ void contiguous_sum_f32(float *out, float *in, size_t size)
{
    all_reduce<
        ContiguousIndexCalculator<float>, float, float, Sum, 256, 32>(out, size, ContiguousIndexCalculator<float>(in));
}

extern "C" __global__ void contiguous_sum_fast_dim_include_f32(float *out, float *in, float *buffer, int *finished, FastDivmod *fd, int *strides, size_t ndim, size_t fast_dim_size, size_t reduce_size_no_fast_dim)
{
    switch (blockDim.x * blockDim.y)
    {
    case 32:
        reduce_fast_dim_include<float, float, Sum, 32, 32>(out, in, buffer, finished, fd, strides, ndim, fast_dim_size, reduce_size_no_fast_dim);
        break;
    case 64:
        reduce_fast_dim_include<float, float, Sum, 64, 32>(out, in, buffer, finished, fd, strides, ndim, fast_dim_size, reduce_size_no_fast_dim);
        break;
    case 128:
        reduce_fast_dim_include<float, float, Sum, 128, 32>(out, in, buffer, finished, fd, strides, ndim, fast_dim_size, reduce_size_no_fast_dim);
        break;
    case 256:
        reduce_fast_dim_include<float, float, Sum, 256, 32>(out, in, buffer, finished, fd, strides, ndim, fast_dim_size, reduce_size_no_fast_dim);
        break;
    case 512:
        reduce_fast_dim_include<float, float, Sum, 512, 32>(out, in, buffer, finished, fd, strides, ndim, fast_dim_size, reduce_size_no_fast_dim);
        break;
    default:
        break;
    }
}

__launch_bounds__(512, 4) extern "C" __global__ void contiguous_sum_fast_dim_only_f32(float *out, float *in, float *buffer, int32_t *finished, size_t fast_dim_size, size_t output_size)
{
#define ARMS(BLOCK_SIZE)                                                                                                \
    case BLOCK_SIZE:                                                                                                    \
        reduce_fast_dim_only<float, float, Sum, BLOCK_SIZE, 32>(out, in, buffer, finished, fast_dim_size, output_size); \
        break;

    switch (blockDim.x)
    {
        ARMS(32);
        ARMS(64);
        ARMS(96);
        ARMS(128);
        ARMS(160);
        ARMS(192);
        ARMS(224);
        ARMS(256);
        ARMS(288);
        ARMS(320);
        ARMS(352);
        ARMS(384);
        ARMS(416);
        ARMS(448);
        ARMS(480);
        ARMS(512);
    default:
        break;
    }
}

__launch_bounds__(512, 4) extern "C" __global__ void contiguous_sum_small_fast_dim_only_f32(float *out, float *in, size_t fast_dim_size, size_t output_size)
{
#define ARMS(BLOCK_SIZE)                                                                                    \
    case BLOCK_SIZE:                                                                                        \
        reduce_small_fast_dim_only<float, float, Sum, BLOCK_SIZE, 32>(out, in, fast_dim_size, output_size); \
        break;

    switch (blockDim.x)
    {
        ARMS(32);
        ARMS(64);
        ARMS(96);
        ARMS(128);
        ARMS(160);
        ARMS(192);
        ARMS(224);
        ARMS(256);
        ARMS(288);
        ARMS(320);
        ARMS(352);
        ARMS(384);
        ARMS(416);
        ARMS(448);
        ARMS(480);
        ARMS(512);
    default:
        break;
    }
}

extern "C" __global__ void contiguous_sum_fast_dim_no_include_f32(float *out, float *buffer, float *in, int *finished, FastDivmod *shape, int *strides, size_t ndim, size_t reduce_size, size_t output_size)
{
    reduce_fast_dim_not_include<float, float, Sum, 32>(out, buffer, in, finished, shape, strides, ndim, reduce_size, output_size);
}