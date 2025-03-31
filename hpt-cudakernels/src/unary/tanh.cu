#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(tanh_f16, f16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_bf16, bf16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_f32, f32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_f64, f64, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_bool, bool, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i8, i8, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i16, i16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i32, i32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_i64, i64, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u8, u8, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u16, u16, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u32, u32, FloatOutUnaryPromote, Tanh);
DEFINE_UNARY_KERNEL(tanh_u64, u64, FloatOutUnaryPromote, Tanh);
