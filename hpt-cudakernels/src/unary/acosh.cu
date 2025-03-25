#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct ACosh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(acosh(cast<Input, f32>(a)));
        }
        else
        {
            return acosh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(acosh_f16, f16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_bf16, bf16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_f32, f32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_f64, f64, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_bool, bool, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i8, i8, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i16, i16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i32, i32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i64, i64, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u8, u8, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u16, u16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u32, u32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u64, u64, FloatOutUnaryPromote, ACosh);
