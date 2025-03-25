#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Log2
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(log2(cast<Input, f32>(a)));
        }
        else
        {
            return log2(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(log2_f16, f16, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_bf16, bf16, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_f32, f32, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_f64, f64, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_bool, bool, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_i8, i8, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_i16, i16, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_i32, i32, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_i64, i64, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_u8, u8, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_u16, u16, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_u32, u32, FloatOutUnaryPromote, Log2);
DEFINE_UNARY_KERNEL(log2_u64, u64, FloatOutUnaryPromote, Log2);
