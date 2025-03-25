#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Ln
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(log(cast<Input, f32>(a)));
        }
        else
        {
            return log(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(ln_f16, f16, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_bf16, bf16, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_f32, f32, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_f64, f64, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_bool, bool, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_i8, i8, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_i16, i16, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_i32, i32, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_i64, i64, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_u8, u8, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_u16, u16, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_u32, u32, FloatOutUnaryPromote, Ln);
DEFINE_UNARY_KERNEL(ln_u64, u64, FloatOutUnaryPromote, Ln);
