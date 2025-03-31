#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(acosh_f16, f16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_bf16, bf16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_f32, f32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_f64, f64, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_bool, bool, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i8, i8, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i16, i16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i32, i32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_i64, i64, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u8, u8, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u16, u16, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u32, u32, FloatOutUnaryPromote, ACosh);
DEFINE_UNARY_KERNEL(acosh_u64, u64, FloatOutUnaryPromote, ACosh);
