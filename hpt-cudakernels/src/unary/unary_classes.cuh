#pragma once
#include "../utils/check_type.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/type_cast.cuh"

template <typename Input>
struct ACos
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(acos(cast<Input, f32>(a)));
        }
        else
        {
            return acos(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct ACosh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(acosh(cast<Input, f32>(a)));
        }
        else
        {
            return acosh(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct ASin
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(asin(cast<Input, f32>(a)));
        }
        else
        {
            return asin(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct ASinh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(asinh(cast<Input, f32>(a)));
        }
        else
        {
            return asinh(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct ATan
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(atan(cast<Input, f32>(a)));
        }
        else
        {
            return atan(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct ATanh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(atanh(cast<Input, f32>(a)));
        }
        else
        {
            return atanh(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Cbrt
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(cbrt(x));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return cbrt(x);
        }
    }
};

template <typename Input>
struct Celu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a, Output scale) const
    {
        CHECK_FLOAT_TYPE(Output);
        return celu(cast<Input, Output>(a), scale);
    }
};

template <typename Input>
struct Cos
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(cos(cast<Input, f32>(a)));
        }
        else
        {
            return cos(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Cosh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(cosh(cast<Input, f32>(a)));
        }
        else
        {
            return cosh(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Elu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a, Output alpha) const
    {
        CHECK_FLOAT_TYPE(Output);
        return elu(cast<Input, Output>(a), alpha);
    }
};

template <typename Input>
struct Erf
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(erff(cast<Input, f32>(a)));
        }
        else
        {
            return erf(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Exp
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(exp(cast<Input, f32>(a)));
        }
        else if constexpr (std::is_same_v<Input, f32>)
        {
            return exp(cast<Input, Output>(a));
        }
        else if constexpr (std::is_same_v<Input, f64>)
        {
            return exp((f64)(a));
        }
        else
        {
            return exp(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Exp2
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(exp2(cast<Input, f32>(a)));
        }
        else
        {
            return exp2(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Exp10
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(exp10(cast<Input, f32>(a)));
        }
        else
        {
            return exp10(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Gelu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16> || std::is_same_v<Output, f32> || std::is_same_v<Input, f32>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(0.5f * x * (erff(x * 0.707106781186547524400844362104849039f) + 1.0f));
        }
        else
        {
            return 0.5 * a * (erf(a * 0.707106781186547524400844362104849039) + 1.0);
        }
    }
};

template <typename Input>
struct HardSigmoid
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            constexpr f32 v = 1.0 / 6.0;
            return cast<f32, Output>(max(min(x * v + 0.5f, 1.0f), 0.0f));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            constexpr Output v = 1.0 / 6.0;
            return max(min(x * v + 0.5, 1.0), 0.0);
        }
    }
};

template <typename Input>
struct HardSwish
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(x * min(max(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return x * min(max(x + 3.0, 0.0), 6.0) * (1.0 / 6.0);
        }
    }
};

template <typename Input>
struct Ln
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(log(cast<Input, f32>(a)));
        }
        else
        {
            return log(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Log2
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(log2(cast<Input, f32>(a)));
        }
        else
        {
            return log2(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Log10
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(log10(cast<Input, f32>(a)));
        }
        else
        {
            return log10(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Mish
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(x * tanh(log(1.0f + expf(x))));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return x * tanh(log(1.0 + exp(x)));
        }
    }
};

template <typename Input>
struct Recip
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(1.0f / cast<Input, f32>(a));
        }
        else
        {
            return cast<f32, Output>(1.0f) / cast<Input, Output>(a);
        }
    }
};

template <typename Input>
struct Selu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            constexpr f32 alpha = 1.6732632423543772848170429916717;
            constexpr f32 gamma = 1.0507009873554804934193349852946;
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(gamma * (max(0.0f, x) + alpha * min(expm1(x), 0.0f)));
        }
        else
        {
            constexpr f64 alpha = 1.6732632423543772848170429916717;
            constexpr f64 gamma = 1.0507009873554804934193349852946;
            f64 x = cast<Input, f64>(a);
            return gamma * (max(0.0, x) + alpha * min(expm1(x), 0.0));
        }
    }
};

template <typename Input>
struct Sigmoid
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(1.0f / (1.0f + expf(-cast<Input, f32>(a))));
        }
        else
        {
            return 1.0 / (1.0 + exp(-cast<Input, Output>(a)));
        }
    }
};

template <typename Input>
struct Sin
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(sin(cast<Input, f32>(a)));
        }
        else
        {
            return sin(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Sinh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(sinh(cast<Input, f32>(a)));
        }
        else
        {
            return sinh(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct SoftPlus
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(log(1.0f + expf(x)));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return log(1.0 + exp(x));
        }
    }
};

template <typename Input>
struct SoftSign
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(x / (1.0f + abs(x)));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return x / (1.0 + abs(x));
        }
    }
};

template <typename Input>
struct Sqrt
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(sqrt(cast<Input, f32>(a)));
        }
        else
        {
            return sqrt(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Rsqrt
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(rsqrt(cast<Input, f32>(a)));
        }
        else
        {
            return rsqrt(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Tan
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(tan(cast<Input, f32>(a)));
        }
        else
        {
            return tan(cast<Input, Output>(a));
        }
    }
};

template <typename Input>
struct Tanh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(tanh(cast<Input, f32>(a)));
        }
        else
        {
            return tanh(cast<Input, Output>(a));
        }
    }
};