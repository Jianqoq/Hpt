#pragma once
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"

template <typename LHS, typename RHS>
struct Add
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a || b;
        }
        else
        {
            return cast<LHS, Output>(a) + cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Bitand
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a & b;
        }
        else
        {
            return cast<LHS, Output>(a) & cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Bitor
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a | b;
        }
        else
        {
            return cast<LHS, Output>(a) | cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Bitxor
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a ^ b;
        }
        else
        {
            return cast<LHS, Output>(a) ^ cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Div
{
    using Output = typename FloatOutBinaryPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a && b;
        }
        else
        {
            return cast<LHS, Output>(a) / cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Mul
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a && b;
        }
        else
        {
            return cast<LHS, Output>(a) * cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Rem
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_floating_point_v<Output> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            if constexpr (std::is_same_v<Output, f16>)
            {
                return cast<f32, f16>(fmodf(cast<LHS, f32>(a), cast<RHS, f32>(b)));
            }
            else if constexpr (std::is_same_v<Output, bf16>)
            {
                return cast<f32, bf16>(fmodf(cast<LHS, f32>(a), cast<RHS, f32>(b)));
            }
            else if constexpr (std::is_same_v<Output, f32>)
            {
                return fmodf(cast<LHS, f32>(a), cast<RHS, f32>(b));
            }
            else if constexpr (std::is_same_v<Output, f64>)
            {
                return fmod(cast<LHS, f64>(a), cast<RHS, f64>(b));
            }
            else
            {
                return fmod(cast<LHS, f64>(a), cast<RHS, f64>(b));
            }
        }
        else
        {
            return cast<LHS, Output>(a) % cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Shl
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a << b;
        }
        else
        {
            return cast<LHS, Output>(a) << cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Shr
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a >> b;
        }
        else
        {
            return cast<LHS, Output>(a) >> cast<RHS, Output>(b);
        }
    }
};

template <typename LHS, typename RHS>
struct Sub
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a && b;
        }
        else
        {
            return cast<LHS, Output>(a) - cast<RHS, Output>(b);
        }
    }
};