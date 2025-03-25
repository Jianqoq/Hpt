#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Gelu
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16> || std::is_same_v<Output, f32> || std::is_same_v<Input, f32>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(0.5f * x * (erff(x * 0.707106781186547524400844362104849039f) + 1.0f));
        }
        else
        {
            return 0.5 * a * (erf(a * 0.707106781186547524400844362104849039) + 1.0);
        }
    }
};

DEFINE_UNARY_KERNEL(gelu_f16, f16, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_bf16, bf16, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_f32, f32, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_f64, f64, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_bool, bool, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_i8, i8, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_i16, i16, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_i32, i32, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_i64, i64, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_u8, u8, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_u16, u16, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_u32, u32, FloatOutUnaryPromote, Gelu);
DEFINE_UNARY_KERNEL(gelu_u64, u64, FloatOutUnaryPromote, Gelu);
