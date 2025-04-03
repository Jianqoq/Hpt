#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(atanh_f16, f16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_bf16, bf16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_f32, f32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_f64, f64, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_bool, bool, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i8, i8, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i16, i16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i32, i32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_i64, i64, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u8, u8, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u16, u16, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u32, u32, FloatOutUnaryPromote, ATanh);
DEFINE_UNARY_KERNEL(atanh_u64, u64, FloatOutUnaryPromote, ATanh);
