#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

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

DEFINE_UNARY_KERNEL(mish_f16, f16, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_bf16, bf16, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_f32, f32, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_f64, f64, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_bool, bool, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_i8, i8, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_i16, i16, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_i32, i32, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_i64, i64, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_u8, u8, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_u16, u16, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_u32, u32, FloatOutUnaryPromote, Mish);
DEFINE_UNARY_KERNEL(mish_u64, u64, FloatOutUnaryPromote, Mish);
