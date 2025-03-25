#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Sin
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(sin(cast<Input, f32>(a)));
        }
        else
        {
            return sin(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(sin_f16, f16, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_bf16, bf16, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_f32, f32, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_f64, f64, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_bool, bool, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_i8, i8, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_i16, i16, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_i32, i32, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_i64, i64, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_u8, u8, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_u16, u16, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_u32, u32, FloatOutUnaryPromote, Sin);
DEFINE_UNARY_KERNEL(sin_u64, u64, FloatOutUnaryPromote, Sin);

