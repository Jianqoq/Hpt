#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(sigmoid_f16, f16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_bf16, bf16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_f32, f32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_f64, f64, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_bool, bool, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i8, i8, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i16, i16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i32, i32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_i64, i64, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u8, u8, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u16, u16, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u32, u32, FloatOutUnaryPromote, Sigmoid);
DEFINE_UNARY_KERNEL(sigmoid_u64, u64, FloatOutUnaryPromote, Sigmoid);
