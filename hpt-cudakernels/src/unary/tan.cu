#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Tan
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(tan(cast<Input, f32>(a)));
        }
        else
        {
            return tan(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(tan_f16, f16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_bf16, bf16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_f32, f32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_f64, f64, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_bool, bool, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i8, i8, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i16, i16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i32, i32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i64, i64, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u8, u8, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u16, u16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u32, u32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u64, u64, FloatOutUnaryPromote, Tan);
