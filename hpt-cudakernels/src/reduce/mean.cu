#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/check_type.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class Mean : public ReduceOp<T, R, WarpSize>
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
        if constexpr (std::is_same_v<T, float>)
        {
            return 0.0f;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return 0.0;
        }
        else if constexpr (std::is_same_v<T, __half>)
        {
            return __float2half(0.0f);
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            return __float2bfloat16(0.0f);
        }
        else
        {
            return T(0);
        }
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a += __shfl_down_sync(0xffffffff, a, mask);
        }
        return a;
    }

    __device__ __forceinline__ static R pre_op(T a)
    {
        CHECK_FLOAT_TYPE(R);
        return cast<T, R>(a);
    }

    __device__ __forceinline__ static R post_op(R a, size_t count)
    {
        if constexpr (std::is_same_v<R, half>)
        {
            return __float2half(__half2float(a) / (float)count);
        }
        else if constexpr (std::is_same_v<R, __nv_bfloat16>)
        {
            return __float2bfloat16(__bfloat162float(a) / (float)count);
        }
        else
        {
            return a / static_cast<R>(count);
        }
    }
};

DECLARE_KERNEL(half, bool, bool, mean, Mean)
DECLARE_KERNEL(half, uint8_t, u8, mean, Mean)
DECLARE_KERNEL(half, uint16_t, u16, mean, Mean)
DECLARE_KERNEL(float, uint32_t, u32, mean, Mean)
DECLARE_KERNEL(double, uint64_t, u64, mean, Mean)
DECLARE_KERNEL(half, int8_t, i8, mean, Mean)
DECLARE_KERNEL(half, int16_t, i16, mean, Mean)
DECLARE_KERNEL(float, int32_t, i32, mean, Mean)
DECLARE_KERNEL(double, int64_t, i64, mean, Mean)
DECLARE_KERNEL(float, float, f32, mean, Mean)
DECLARE_KERNEL(double, double, f64, mean, Mean)
DECLARE_KERNEL(half, half, f16, mean, Mean)
DECLARE_KERNEL(__nv_bfloat16, __nv_bfloat16, bf16, mean, Mean)
