#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Cbrt
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16> || std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            f32 x = cast<Input, f32>(a);
            return cast<f32, Output>(cbrt(x));
        }
        else
        {
            Output x = cast<Input, Output>(a);
            return cbrt(x);
        }
    }
};

DEFINE_UNARY_KERNEL(cbrt_f16, f16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_bf16, bf16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_f32, f32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_f64, f64, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_bool, bool, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i8, i8, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i16, i16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i32, i32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i64, i64, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u8, u8, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u16, u16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u32, u32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u64, u64, FloatOutUnaryPromote, Cbrt);
