#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(soft_plus_f16, f16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_bf16, bf16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_f32, f32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_f64, f64, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_bool, bool, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i8, i8, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i16, i16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i32, i32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_i64, i64, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u8, u8, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u16, u16, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u32, u32, FloatOutUnaryPromote, SoftPlus);
DEFINE_UNARY_KERNEL(soft_plus_u64, u64, FloatOutUnaryPromote, SoftPlus);

