#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(cosh_f16, f16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_bf16, bf16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_f32, f32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_f64, f64, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_bool, bool, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i8, i8, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i16, i16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i32, i32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_i64, i64, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u8, u8, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u16, u16, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u32, u32, FloatOutUnaryPromote, Cosh);
DEFINE_UNARY_KERNEL(cosh_u64, u64, FloatOutUnaryPromote, Cosh);
