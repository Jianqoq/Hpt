#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(recip_f16, f16, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_bf16, bf16, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_f32, f32, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_f64, f64, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_bool, bool, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_i8, i8, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_i16, i16, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_i32, i32, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_i64, i64, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_u8, u8, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_u16, u16, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_u32, u32, FloatOutUnaryPromote, Recip);
DEFINE_UNARY_KERNEL(recip_u64, u64, FloatOutUnaryPromote, Recip);

