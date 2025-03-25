#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Sqrt
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(sqrt(cast<Input, f32>(a)));
        }
        else
        {
            return sqrt(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(sqrt_f16, f16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_bf16, bf16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_f32, f32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_f64, f64, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_bool, bool, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i8, i8, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i16, i16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i32, i32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i64, i64, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u8, u8, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u16, u16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u32, u32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u64, u64, FloatOutUnaryPromote, Sqrt);
