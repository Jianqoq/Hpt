#include "reduce/reduce_template.cuh"
#include "reduce/reduce_helper.cuh"
#include "utils/loop_progress.cuh"

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

extern "C" __global__ void contiguous_sum_dim_include_f32(float *out, float *in, long long *shape, long long *strides, size_t ndim, size_t fast_dim_size, size_t num_elements_per_thread, size_t reduce_ndim_exclude_fast_dim)
{
    long long prg[25];
    auto func = [ndim, shape, strides, reduce_ndim_exclude_fast_dim, &prg] __device__(float *&data)
    {
        for (size_t i = ndim - 1; i > ndim - reduce_ndim_exclude_fast_dim; i--)
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
    reduce_fast_dim_include<float, float>(out, in, shape, strides, ndim, fast_dim_size, num_elements_per_thread, progress_updater);
}