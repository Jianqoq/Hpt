#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(exp_f16, f16, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_bf16, bf16, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_f32, f32, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_f64, f64, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_bool, bool, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_i8, i8, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_i16, i16, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_i32, i32, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_i64, i64, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_u8, u8, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_u16, u16, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_u32, u32, FloatOutUnaryPromote, Exp);
DEFINE_UNARY_KERNEL(exp_u64, u64, FloatOutUnaryPromote, Exp);
