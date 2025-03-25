#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Cos
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(cos(cast<Input, f32>(a)));
        }
        else
        {
            return cos(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(cos_f16, f16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_bf16, bf16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_f32, f32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_f64, f64, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_bool, bool, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i8, i8, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i16, i16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i32, i32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i64, i64, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u8, u8, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u16, u16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u32, u32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u64, u64, FloatOutUnaryPromote, Cos);

