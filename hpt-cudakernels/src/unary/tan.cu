#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(tan_f16, f16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_bf16, bf16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_f32, f32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_f64, f64, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_bool, bool, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i8, i8, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i16, i16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i32, i32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_i64, i64, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u8, u8, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u16, u16, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u32, u32, FloatOutUnaryPromote, Tan);
DEFINE_UNARY_KERNEL(tan_u64, u64, FloatOutUnaryPromote, Tan);
