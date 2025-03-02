#include "arg_template.cuh"
#include "../utils/type_utils.cuh"
#include "declare_macros.cuh"
#include <stdio.h>
template <typename T>
struct ArgMaxResult
{
    T val;
    int64_t idx;

    __device__ __forceinline__ ArgMaxResult(T val, int64_t idx) : val(val), idx(idx) {}

    __device__ __forceinline__ static ArgMaxResult identity()
    {
        return ArgMaxResult();
    }

    __device__ __forceinline__ ArgMaxResult() : val(TypeUtils<T>::limit_min()), idx(0) {}
};

template <typename T, unsigned int WarpSize = 32>
class ArgMax
{
public:
    __device__ __forceinline__ static ArgMaxResult<T> combine(ArgMaxResult<T> a, ArgMaxResult<T> b)
    {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hgt(a.val, b.val) ? a : (__hlt(a.val, b.val) ? b : (a.idx < b.idx ? a : b));
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return a.val > b.val ? a : (a.val < b.val ? b : (a.idx < b.idx ? a : b));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return a.val > b.val ? a : (a.val < b.val ? b : (a.idx < b.idx ? a : b));
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a.val > b.val ? a : (a.val < b.val ? b : (a.idx < b.idx ? a : b));
        }
        else
        {
            return a.val > b.val ? a : (a.val < b.val ? b : (a.idx < b.idx ? a : b));
        }
    }

    __device__ __forceinline__ static ArgMaxResult<T> identity()
    {
        return ArgMaxResult<T>::identity();
    }

    __device__ __forceinline__ static ArgMaxResult<T> warp_reduce(ArgMaxResult<T> a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            T value = __shfl_down_sync(0xffffffff, a.val, mask);
            int64_t index = __shfl_down_sync(0xffffffff, a.idx, mask);
            ArgMaxResult<T> other{value, index};
            a = ArgMax<T, WarpSize>::combine(a, other);
        }
        return a;
    }
};

DECLARE_ARG_KERNEL(bool, bool, argmax, ArgMax);
DECLARE_ARG_KERNEL(int8_t, i8, argmax, ArgMax);
DECLARE_ARG_KERNEL(int16_t, i16, argmax, ArgMax);
DECLARE_ARG_KERNEL(int32_t, i32, argmax, ArgMax);
DECLARE_ARG_KERNEL(int64_t, i64, argmax, ArgMax);
DECLARE_ARG_KERNEL(uint8_t, u8, argmax, ArgMax);
DECLARE_ARG_KERNEL(uint16_t, u16, argmax, ArgMax);
DECLARE_ARG_KERNEL(uint32_t, u32, argmax, ArgMax);
DECLARE_ARG_KERNEL(uint64_t, u64, argmax, ArgMax);
DECLARE_ARG_KERNEL(__half, f16, argmax, ArgMax);
DECLARE_ARG_KERNEL(float, f32, argmax, ArgMax);
DECLARE_ARG_KERNEL(double, f64, argmax, ArgMax);
DECLARE_ARG_KERNEL(__nv_bfloat16, bf16, argmax, ArgMax);
