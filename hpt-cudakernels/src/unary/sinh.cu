#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Sinh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(sinh(cast<Input, f32>(a)));
        }
        else
        {
            return sinh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(sinh_f16, f16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_bf16, bf16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_f32, f32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_f64, f64, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_bool, bool, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i8, i8, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i16, i16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i32, i32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i64, i64, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u8, u8, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u16, u16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u32, u32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u64, u64, FloatOutUnaryPromote, Sinh);

