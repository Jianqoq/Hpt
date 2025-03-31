#pragma once
#include "../utils/promotion/promotes.cuh"
#include "reduce_helper.cuh"
#include "../utils/type_utils.cuh"

template <typename T, unsigned int WarpSize = 32>
class Max : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
    __device__ __forceinline__ static T combine(T a, T b)
    {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hmax(a, b);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return fmaxf(a, b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return fmax(a, b);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a || b;
        }
        else
        {
            return (a > b) ? a : b;
        }
    }

    __device__ __forceinline__ static T identity()
    {
        return TypeUtils<T>::limit_min();
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a = Max<R, WarpSize>::combine(a, __shfl_down_sync(0xffffffff, a, mask));
        }
        return a;
    }
};

template <typename T, unsigned int WarpSize = 32>
class Mean : public ReduceOp<T, FloatOutBinaryPromote<T, T>::Output, WarpSize>
{
public:
    using R = FloatOutBinaryPromote<T, T>::Output;
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

template <typename T, unsigned int WarpSize = 32>
class All : public ReduceOp<T, bool, WarpSize>
{
public:
    using R = bool;
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

template <typename T, unsigned int WarpSize = 32>
class Any : public ReduceOp<T, bool, WarpSize>
{
public:
    using R = bool;
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

template <typename T, unsigned int WarpSize = 32>
class LogSumExp : public ReduceOp<T, FloatOutBinaryPromote<T, T>::Output, WarpSize>
{
public:
    using R = FloatOutBinaryPromote<T, T>::Output;
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
        if constexpr (std::is_same_v<R, half>)
        {
            return __float2half(expf(cast<T, float>(a)));
        }
        else if constexpr (std::is_same_v<R, __nv_bfloat16>)
        {
            return __float2bfloat16(expf(cast<T, float>(a)));
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            return expf(cast<T, float>(a));
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            return exp(cast<T, double>(a));
        }
        else
        {
            return cast<T, R>(a);
        }
    }

    __device__ __forceinline__ static R post_op(R a, size_t count)
    {
        CHECK_FLOAT_TYPE(R);
        if constexpr (std::is_same_v<R, half>)
        {
            return __float2half(logf(__half2float(a)));
        }
        else if constexpr (std::is_same_v<R, __nv_bfloat16>)
        {
            return __float2bfloat16(logf(__bfloat162float(a)));
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            return logf(a);
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            return log(a);
        }
        else
        {
            return a;
        }
    }
};

template <typename T, unsigned int WarpSize = 32>
class Min : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
    __device__ __forceinline__ static T combine(T a, T b)
    {
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return __hmin(a, b);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return fminf(a, b);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return fmin(a, b);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a && b;
        }
        else
        {
            return (a < b) ? a : b;
        }
    }

    __device__ __forceinline__ static T identity()
    {
        return TypeUtils<T>::limit_max();
    }

    __device__ __forceinline__ static R warp_reduce(R a)
    {
#pragma unroll
        for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1)
        {
            a = Min<R, WarpSize>::combine(a, __shfl_down_sync(0xffffffff, a, mask));
        }
        return a;
    }
};

template <typename T, unsigned int WarpSize = 32>
class NanProd : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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

template <typename T, unsigned int WarpSize = 32>
class NanSum : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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

template <typename T, unsigned int WarpSize = 32>
class Prod : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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

template <typename T, unsigned int WarpSize = 32>
class ReduceL1 : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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
        R res = cast<T, R>(a);
        if constexpr (std::is_same_v<R, half> || std::is_same_v<R, __nv_bfloat16>)
        {
            return __habs(res);
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            return fabsf(res);
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            return fabs(res);
        }
        else if constexpr (std::is_signed_v<R>)
        {
            return res < 0 ? -res : res;
        }
        else
        {
            return res;
        }
    }
};

template <typename T, unsigned int WarpSize = 32>
class ReduceL2 : public ReduceOp<T, FloatOutBinaryPromote<T, T>::Output, WarpSize>
{
public:
    using R = FloatOutBinaryPromote<T, T>::Output;
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
        CHECK_FLOAT_TYPE(R);
        R res = cast<T, R>(a);
        return res * res;
    }

    __device__ __forceinline__ static R post_op(R a, size_t count)
    {
        CHECK_FLOAT_TYPE(R);
        if constexpr (std::is_same_v<R, half>)
        {
            return __float2half(sqrtf(__half2float(a)));
        }
        else if constexpr (std::is_same_v<R, __nv_bfloat16>)
        {
            return __float2bfloat16(sqrtf(__bfloat162float(a)));
        }
        else if constexpr (std::is_same_v<R, float>)
        {
            return sqrtf(a);
        }
        else if constexpr (std::is_same_v<R, double>)
        {
            return sqrt(a);
        }
        else
        {
            return a;
        }
    }
};

template <typename T, unsigned int WarpSize = 32>
class ReduceL3 : public ReduceOp<T, FloatOutBinaryPromote<T, T>::Output, WarpSize>
{
public:
    using R = FloatOutBinaryPromote<T, T>::Output;
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

template <typename T, unsigned int WarpSize = 32>
class SumSquare : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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
            return __hmul(a, a);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return a && a;
        }
        else
        {
            return a * a;
        }
    }
};

template <typename T, unsigned int WarpSize = 32>
class Sum : public ReduceOp<T, T, WarpSize>
{
public:
    using R = T;
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
};