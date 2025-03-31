#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

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
