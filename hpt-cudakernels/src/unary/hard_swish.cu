#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct HardSwish
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(x * min(max(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return x * min(max(x + 3.0, 0.0), 6.0) * (1.0 / 6.0);
        }
    }
};

DEFINE_UNARY_KERNEL(hard_swish_f16, f16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_bf16, bf16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_f32, f32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_f64, f64, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_bool, bool, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i8, i8, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i16, i16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i32, i32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i64, i64, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u8, u8, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u16, u16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u32, u32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u64, u64, FloatOutUnaryPromote, HardSwish);
