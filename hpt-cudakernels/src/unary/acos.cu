#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct ACos
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(acos(cast<Input, f32>(a)));
        }
        else
        {
            return acos(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(acos_f16, f16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_bf16, bf16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_f32, f32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_f64, f64, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_bool, bool, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i8, i8, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i16, i16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i32, i32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i64, i64, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u8, u8, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u16, u16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u32, u32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u64, u64, FloatOutUnaryPromote, ACos);
