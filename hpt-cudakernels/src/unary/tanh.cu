#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Tanh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(tanh(cast<Input, f32>(a)));
        }
        else
        {
            return tanh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(tanh_f16, f16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_bf16, bf16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_f32, f32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_f64, f64, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_bool, bool, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i8, i8, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i16, i16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i32, i32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i64, i64, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u8, u8, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u16, u16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u32, u32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u64, u64, FloatOutUnaryPromote, Tanh);
