#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Cosh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(cosh(cast<Input, f32>(a)));
        }
        else
        {
            return cosh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(cosh_f16, f16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_bf16, bf16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_f32, f32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_f64, f64, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_bool, bool, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i8, i8, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i16, i16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i32, i32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i64, i64, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u8, u8, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u16, u16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u32, u32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u64, u64, FloatOutUnaryPromote, Cosh);
