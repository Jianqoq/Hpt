#include "declare_macros.cuh"
#include "../type_cast.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class Any : public ReduceOp<T, R, WarpSize>
{
public:
    __device__ __forceinline__ static T combine(T a, T b)
    {
        return a || b;
    }

    __device__ __forceinline__ static T identity()
    {
        return false;
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a |= __shfl_down_sync(0xffffffff, a, mask);
        }
        return a;
    }

    __device__ __forceinline__ static R pre_op(T a)
    {
        static_assert(std::is_same_v<R, bool>, "Any only supports bool");
        return cast<T, R>(a);
    }
};

DECLARE_KERNEL(bool, bool, bool, any, Any)
DECLARE_KERNEL(bool, uint8_t, u8, any, Any)
DECLARE_KERNEL(bool, uint16_t, u16, any, Any)
DECLARE_KERNEL(bool, uint32_t, u32, any, Any)
DECLARE_KERNEL(bool, uint64_t, u64, any, Any)
DECLARE_KERNEL(bool, int8_t, i8, any, Any)
DECLARE_KERNEL(bool, int16_t, i16, any, Any)
DECLARE_KERNEL(bool, int32_t, i32, any, Any)
DECLARE_KERNEL(bool, int64_t, i64, any, Any)
DECLARE_KERNEL(bool, float, f32, any, Any)
DECLARE_KERNEL(bool, double, f64, any, Any)
DECLARE_KERNEL(bool, __half, f16, any, Any)
DECLARE_KERNEL(bool, __nv_bfloat16, bf16, any, Any)
