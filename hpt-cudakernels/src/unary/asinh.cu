#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct ASinh
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(asinh(cast<Input, f32>(a)));
        }
        else
        {
            return asinh(cast<Input, Output>(a));
        }
    }
};

DEFINE_UNARY_KERNEL(asinh_f16, f16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_bf16, bf16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_f32, f32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_f64, f64, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_bool, bool, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i8, i8, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i16, i16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i32, i32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i64, i64, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u8, u8, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u16, u16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u32, u32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u64, u64, FloatOutUnaryPromote, ASinh);
