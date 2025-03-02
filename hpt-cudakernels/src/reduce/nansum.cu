#include "declare_macros.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class NanSum : public ReduceOp<T, R, WarpSize>
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
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hisnan(a) ? identity() : a;
        }
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return isnan(a) ? identity() : a;
        }
        else
        {
            return a;
        }
    }
};

DECLARE_KERNEL(bool, bool, bool, nansum, NanSum)
DECLARE_KERNEL(uint8_t, uint8_t, u8, nansum, NanSum)
DECLARE_KERNEL(uint16_t, uint16_t, u16, nansum, NanSum)
DECLARE_KERNEL(uint32_t, uint32_t, u32, nansum, NanSum)
DECLARE_KERNEL(uint64_t, uint64_t, u64, nansum, NanSum)
DECLARE_KERNEL(int8_t, int8_t, i8, nansum, NanSum)
DECLARE_KERNEL(int16_t, int16_t, i16, nansum, NanSum)
DECLARE_KERNEL(int32_t, int32_t, i32, nansum, NanSum)
DECLARE_KERNEL(int64_t, int64_t, i64, nansum, NanSum)
DECLARE_KERNEL(float, float, f32, nansum, NanSum)
DECLARE_KERNEL(double, double, f64, nansum, NanSum)
DECLARE_KERNEL(__half, __half, f16, nansum, NanSum)
DECLARE_KERNEL(__nv_bfloat16, __nv_bfloat16, bf16, nansum, NanSum)
