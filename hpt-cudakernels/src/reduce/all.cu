#include "declare_macros.cuh"
#include "../type_cast.cuh"

template <typename T, typename R, unsigned int WarpSize = 32>
class All : public ReduceOp<T, R, WarpSize>
{
public:
    __device__ __forceinline__ static T combine(T a, T b)
    {
        return a && b;
    }

    __device__ __forceinline__ static T identity()
    {
        return true;
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a &= __shfl_down_sync(0xffffffff, a, mask);
        }
        return a;
    }

    __device__ __forceinline__ static R pre_op(T a)
    {
        static_assert(std::is_same_v<R, bool>, "All only supports bool");
        return cast<T, R>(a);
    }
};

DECLARE_KERNEL(bool, bool, bool, all, All)
DECLARE_KERNEL(bool, uint8_t, u8, all, All)
DECLARE_KERNEL(bool, uint16_t, u16, all, All)
DECLARE_KERNEL(bool, uint32_t, u32, all, All)
DECLARE_KERNEL(bool, uint64_t, u64, all, All)
DECLARE_KERNEL(bool, int8_t, i8, all, All)
DECLARE_KERNEL(bool, int16_t, i16, all, All)
DECLARE_KERNEL(bool, int32_t, i32, all, All)
DECLARE_KERNEL(bool, int64_t, i64, all, All)
DECLARE_KERNEL(bool, float, f32, all, All)
DECLARE_KERNEL(bool, double, f64, all, All)
DECLARE_KERNEL(bool, __half, f16, all, All)
DECLARE_KERNEL(bool, __nv_bfloat16, bf16, all, All)
