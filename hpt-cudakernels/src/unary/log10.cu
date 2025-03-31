#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(log10_f16, f16, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_bf16, bf16, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_f32, f32, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_f64, f64, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_bool, bool, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_i8, i8, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_i16, i16, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_i32, i32, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_i64, i64, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_u8, u8, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_u16, u16, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_u32, u32, FloatOutUnaryPromote, Log10);
DEFINE_UNARY_KERNEL(log10_u64, u64, FloatOutUnaryPromote, Log10);
