#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(soft_sign_f16, f16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_bf16, bf16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_f32, f32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_f64, f64, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_bool, bool, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i8, i8, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i16, i16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i32, i32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_i64, i64, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u8, u8, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u16, u16, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u32, u32, FloatOutUnaryPromote, SoftSign);
DEFINE_UNARY_KERNEL(soft_sign_u64, u64, FloatOutUnaryPromote, SoftSign);

