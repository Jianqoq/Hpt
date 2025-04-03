#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(atan_f16, f16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_bf16, bf16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_f32, f32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_f64, f64, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_bool, bool, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i8, i8, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i16, i16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i32, i32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_i64, i64, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u8, u8, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u16, u16, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u32, u32, FloatOutUnaryPromote, ATan);
DEFINE_UNARY_KERNEL(atan_u64, u64, FloatOutUnaryPromote, ATan);
