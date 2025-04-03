#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(sqrt_f16, f16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_bf16, bf16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_f32, f32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_f64, f64, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_bool, bool, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i8, i8, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i16, i16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i32, i32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_i64, i64, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u8, u8, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u16, u16, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u32, u32, FloatOutUnaryPromote, Sqrt);
DEFINE_UNARY_KERNEL(sqrt_u64, u64, FloatOutUnaryPromote, Sqrt);
