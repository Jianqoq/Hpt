#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(sinh_f16, f16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_bf16, bf16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_f32, f32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_f64, f64, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_bool, bool, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i8, i8, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i16, i16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i32, i32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_i64, i64, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u8, u8, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u16, u16, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u32, u32, FloatOutUnaryPromote, Sinh);
DEFINE_UNARY_KERNEL(sinh_u64, u64, FloatOutUnaryPromote, Sinh);

