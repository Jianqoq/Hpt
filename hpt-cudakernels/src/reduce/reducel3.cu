#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/check_type.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class ReduceL3 : public ReduceOp<T, R, WarpSize>
{
public:
    __device__ __forceinline__ static T combine(T a, T b)
    {
        return a + b;
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
        R res = cast<T, R>(a);
        CHECK_FLOAT_TYPE(R);
        if constexpr (std::is_same_v<R, half> || std::is_same_v<R, __nv_bfloat16>)
        {
            R abs = __habs(res);
            return abs * abs * abs;
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            float abs = fabsf(res);
            return abs * abs * abs;
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            double abs = fabs(res);
            return abs * abs * abs;
        }
        else
        {
            return res;
        }
    }

    __device__ __forceinline__ static R post_op(R a, size_t count)
    {
        if constexpr (std::is_same_v<R, half>)
        {
            return __float2half(cbrtf(__half2float(a)));
        }
        else if constexpr (std::is_same_v<R, __nv_bfloat16>)
        {
            return __float2bfloat16(cbrtf(__bfloat162float(a)));
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            return cbrtf(a);
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            return cbrt(a);
        }
        else
        {
            return a;
        }
    }
};

DECLARE_KERNEL(half, bool, bool, reducel3, ReduceL3)
DECLARE_KERNEL(half, uint8_t, u8, reducel3, ReduceL3)
DECLARE_KERNEL(half, uint16_t, u16, reducel3, ReduceL3)
DECLARE_KERNEL(float, uint32_t, u32, reducel3, ReduceL3)
DECLARE_KERNEL(double, uint64_t, u64, reducel3, ReduceL3)
DECLARE_KERNEL(half, int8_t, i8, reducel3, ReduceL3)
DECLARE_KERNEL(half, int16_t, i16, reducel3, ReduceL3)
DECLARE_KERNEL(float, int32_t, i32, reducel3, ReduceL3)
DECLARE_KERNEL(double, int64_t, i64, reducel3, ReduceL3)
DECLARE_KERNEL(float, float, f32, reducel3, ReduceL3)
DECLARE_KERNEL(double, double, f64, reducel3, ReduceL3)
DECLARE_KERNEL(half, half, f16, reducel3, ReduceL3)
DECLARE_KERNEL(__nv_bfloat16, __nv_bfloat16, bf16, reducel3, ReduceL3)
