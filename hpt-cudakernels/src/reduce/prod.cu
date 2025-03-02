#include "declare_macros.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class Prod : public ReduceOp<T, R, WarpSize>
{
public:
    __device__ __forceinline__ static T combine(T a, T b)
    {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hmul(a, b);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a && b;
        }
        else
        {
            return a * b;
        }
    }

    __device__ __forceinline__ static T identity()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return 1.0f;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return 1.0;
        }
        else if constexpr (std::is_same_v<T, __half>)
        {
            return __float2half(1.0f);
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            return __float2bfloat16(1.0f);
        }
        else
        {
            return T(1);
        }
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a *= __shfl_down_sync(0xffffffff, a, mask);
        }
        return a;
    }
};

DECLARE_KERNEL(bool, bool, bool, prod, Prod)
DECLARE_KERNEL(uint8_t, uint8_t, u8, prod, Prod)
DECLARE_KERNEL(uint16_t, uint16_t, u16, prod, Prod)
DECLARE_KERNEL(uint32_t, uint32_t, u32, prod, Prod)
DECLARE_KERNEL(uint64_t, uint64_t, u64, prod, Prod)
DECLARE_KERNEL(int8_t, int8_t, i8, prod, Prod)
DECLARE_KERNEL(int16_t, int16_t, i16, prod, Prod)
DECLARE_KERNEL(int32_t, int32_t, i32, prod, Prod)
DECLARE_KERNEL(int64_t, int64_t, i64, prod, Prod)
DECLARE_KERNEL(float, float, f32, prod, Prod)
DECLARE_KERNEL(double, double, f64, prod, Prod)
DECLARE_KERNEL(__half, __half, f16, prod, Prod)
DECLARE_KERNEL(__nv_bfloat16, __nv_bfloat16, bf16, prod, Prod)
