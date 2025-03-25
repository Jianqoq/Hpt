#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct SoftPlus
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(log(1.0f + expf(x)));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return log(1.0 + exp(x));
        }
    }
};

DEFINE_UNARY_KERNEL(soft_plus_f16, f16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_bf16, bf16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_f32, f32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_f64, f64, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_bool, bool, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i8, i8, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i16, i16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i32, i32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i64, i64, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u8, u8, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u16, u16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u32, u32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u64, u64, FloatOutUnaryPromote, SoftPlus);
