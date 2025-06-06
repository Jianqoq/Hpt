#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(asinh_f16, f16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_bf16, bf16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_f32, f32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_f64, f64, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_bool, bool, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i8, i8, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i16, i16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i32, i32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_i64, i64, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u8, u8, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u16, u16, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u32, u32, FloatOutUnaryPromote, ASinh);
DEFINE_UNARY_KERNEL(asinh_u64, u64, FloatOutUnaryPromote, ASinh);
