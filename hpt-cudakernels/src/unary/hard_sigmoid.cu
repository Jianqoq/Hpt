#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct HardSigmoid
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            constexpr f32 v = 1.0 / 6.0;
            return cast<f32, Output>(max(min(x * v + 0.5f, 1.0f), 0.0f));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            constexpr Output v = 1.0 / 6.0;
            return max(min(x * v + 0.5, 1.0), 0.0);
        }
    }
};

DEFINE_UNARY_KERNEL(hard_sigmoid_f16, f16, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_bf16, bf16, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_f32, f32, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_f64, f64, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_bool, bool, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_i8, i8, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_i16, i16, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_i32, i32, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_i64, i64, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_u8, u8, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_u16, u16, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_u32, u32, FloatOutUnaryPromote, HardSigmoid);
DEFINE_UNARY_KERNEL(hard_sigmoid_u64, u64, FloatOutUnaryPromote, HardSigmoid);
