#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"

template <typename Input>
struct Sigmoid
{
    using Output = typename FloatOutUnaryPromote<Input>::Output;
    __device__ __forceinline__ Output operator()(Input a) const
    {
        CHECK_FLOAT_TYPE(Output);
        if constexpr (std::is_same_v<Input, f16> || std::is_same_v<Input, bf16>)
        {
            return cast<f32, Output>(1.0f / (1.0f + expf(-cast<Input, f32>(a))));
        }
        else if constexpr (std::is_same_v<Output, f16> || std::is_same_v<Output, bf16>)
        {
            return cast<f32, Output>(1.0f / (1.0f + expf(-cast<Input, f32>(a))));
        }
        else
        {
            return 1.0 / (1.0 + exp(-cast<Input, Output>(a)));
        }
    }
};

DEFINE_UNARY_KERNEL(sigmoid_f16, f16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_bf16, bf16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_f32, f32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_f64, f64, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_bool, bool, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i8, i8, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i16, i16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i32, i32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i64, i64, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u8, u8, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u16, u16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u32, u32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u64, u64, FloatOutUnaryPromote, Sigmoid);
