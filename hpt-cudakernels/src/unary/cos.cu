#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(cos_f16, f16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_bf16, bf16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_f32, f32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_f64, f64, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_bool, bool, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i8, i8, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i16, i16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i32, i32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_i64, i64, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u8, u8, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u16, u16, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u32, u32, FloatOutUnaryPromote, Cos);
DEFINE_UNARY_KERNEL(cos_u64, u64, FloatOutUnaryPromote, Cos);

