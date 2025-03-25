#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct ATan
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(atan(cast<Input, f32>(a)));
        }
        else
        {
            return atan(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(atan_f16, f16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_bf16, bf16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_f32, f32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_f64, f64, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_bool, bool, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i8, i8, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i16, i16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i32, i32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i64, i64, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u8, u8, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u16, u16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u32, u32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u64, u64, FloatOutUnaryPromote, ATan);
