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

extern "C" __global__ void contiguous_sum_fast_dim_include_f32(float *out, float *in, FastDivmod *fd, int *shape, int *strides, size_t ndim, size_t fast_dim_size, size_t num_elements_per_thread, size_t reduce_size_no_fast_dim, size_t reduce_ndim_exclude_fast_dim)
{
    int prg[25];
    auto func = [ndim, shape, strides, reduce_ndim_exclude_fast_dim, &prg] __device__(float *&data)
    {
        for (int i = ndim - 1; i >= ndim - reduce_ndim_exclude_fast_dim; i--)
        {
            if (prg[i] < shape[i] - 1)
            {
                prg[i]++;
                data += strides[i];
                break;
            }
            else
            {
                prg[i] = 0;
                data -= strides[i] * (shape[i] - 1);
            }
        }
    };
    auto progress_updater = ProgressUpdater<float, decltype(func)>(func, in);
    if (reduce_ndim_exclude_fast_dim == 1)
    {
        reduce_fast_dim_include<float, float, Sum, ProgressUpdater<float, decltype(func)>, 32, true>(out, in, fd, strides, ndim, fast_dim_size, num_elements_per_thread, reduce_size_no_fast_dim, progress_updater);
    }
    else
    {
        reduce_fast_dim_include<float, float, Sum, ProgressUpdater<float, decltype(func)>, 32, false>(out, in, fd, strides, ndim, fast_dim_size, num_elements_per_thread, reduce_size_no_fast_dim, progress_updater);
    }
}

__launch_bounds__(512, 4) extern "C" __global__ void contiguous_sum_fast_dim_only_f32(float *out, float *in, size_t fast_dim_size, size_t output_size)
{
    switch (blockDim.x)
    {
    case 32:
        reduce_fast_dim_only<float, float, Sum, 32, 32>(out, in, fast_dim_size, output_size);
        break;
    case 64:
        reduce_fast_dim_only<float, float, Sum, 64, 32>(out, in, fast_dim_size, output_size);
        break;
    case 128:
        reduce_fast_dim_only<float, float, Sum, 128, 32>(out, in, fast_dim_size, output_size);
        break;
    case 256:
        reduce_fast_dim_only<float, float, Sum, 256, 32>(out, in, fast_dim_size, output_size);
        break;
    case 512:
        reduce_fast_dim_only<float, float, Sum, 512, 32>(out, in, fast_dim_size, output_size);
        break;
    default:
        break;
    }
}

extern "C" __global__ void contiguous_sum_fast_dim_no_include_f32(float *out, float *in, FastDivmod *shape, int *strides, size_t ndim, size_t reduce_size, size_t reduce_ndim, size_t output_size)
{
    uint64_t prg[25];
    auto func = [ndim, shape, strides, reduce_ndim, &prg] __device__(float *&data)
    {
        for (int i = ndim - 1; i >= ndim - reduce_ndim; i--)
        {
            if (prg[i] < shape[i] - 1)
            {
                prg[i]++;
                data += strides[i];
                break;
            }
            else
            {
                prg[i] = 0;
                data -= strides[i] * (shape[i] - 1);
            }
        }
    };
    auto progress_updater = ProgressUpdater<float, decltype(func)>(func, in);
    if (reduce_ndim == 1)
    {
        reduce_fast_dim_not_include<float, float, Sum, ProgressUpdater<float, decltype(func)>, 32, true>(out, in, shape, strides, ndim, reduce_size, progress_updater, output_size);
    }
    else
    {
        reduce_fast_dim_not_include<float, float, Sum, ProgressUpdater<float, decltype(func)>, 32, false>(out, in, shape, strides, ndim, reduce_size, progress_updater, output_size);
    }
}

extern "C" __global__ void contiguous_sum_fast_dim_no_include_small_f32(
    float *out,
    float *in,
    FastDivmod *shape,
    int *strides,
    size_t ndim,
    size_t reduce_size,
    size_t output_size,
    size_t num_el_per_thread)
{
    reduce_fast_dim_not_include_small<float, float, Sum, 32>(
        out,
        in,
        shape,
        strides,
        ndim,
        reduce_size,
        output_size,
        num_el_per_thread);
}