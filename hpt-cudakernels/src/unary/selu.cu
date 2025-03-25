#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

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

DEFINE_UNARY_KERNEL(selu_f16, f16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_bf16, bf16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_f32, f32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_f64, f64, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_bool, bool, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i8, i8, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i16, i16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i32, i32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i64, i64, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u8, u8, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u16, u16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u32, u32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u64, u64, FloatOutUnaryPromote, Selu);
