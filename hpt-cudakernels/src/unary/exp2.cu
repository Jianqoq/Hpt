#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(exp2_f16, f16, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_bf16, bf16, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_f32, f32, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_f64, f64, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_bool, bool, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_i8, i8, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_i16, i16, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_i32, i32, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_i64, i64, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_u8, u8, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_u16, u16, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_u32, u32, FloatOutUnaryPromote, Exp2);
DEFINE_UNARY_KERNEL(exp2_u64, u64, FloatOutUnaryPromote, Exp2);
