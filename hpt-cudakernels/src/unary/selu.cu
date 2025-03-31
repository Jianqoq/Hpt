#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(selu_f16, f16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_bf16, bf16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_f32, f32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_f64, f64, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_bool, bool, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i8, i8, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i16, i16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i32, i32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_i64, i64, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u8, u8, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u16, u16, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u32, u32, FloatOutUnaryPromote, Selu);
DEFINE_UNARY_KERNEL(selu_u64, u64, FloatOutUnaryPromote, Selu);
