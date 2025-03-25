#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Erf
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(erff(cast<Input, f32>(a)));
        }
        else
        {
            return erf(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(erf_f16, f16, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_bf16, bf16, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_f32, f32, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_f64, f64, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_bool, bool, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_i8, i8, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_i16, i16, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_i32, i32, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_i64, i64, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_u8, u8, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_u16, u16, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_u32, u32, FloatOutUnaryPromote, Erf);
DEFINE_UNARY_KERNEL(erf_u64, u64, FloatOutUnaryPromote, Erf);

