#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

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

DEFINE_UNARY_KERNEL(soft_sign_f16, f16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_bf16, bf16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_f32, f32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_f64, f64, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_bool, bool, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i8, i8, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i16, i16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i32, i32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i64, i64, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u8, u8, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u16, u16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u32, u32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u64, u64, FloatOutUnaryPromote, SoftSign);
