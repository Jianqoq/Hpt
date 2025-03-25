#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct ATanh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(atanh(cast<Input, f32>(a)));
        }
        else
        {
            return atanh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(atanh_f16, f16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_bf16, bf16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_f32, f32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_f64, f64, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_bool, bool, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i8, i8, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i16, i16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i32, i32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i64, i64, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u8, u8, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u16, u16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u32, u32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u64, u64, FloatOutUnaryPromote, ATanh);
