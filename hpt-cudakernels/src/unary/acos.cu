#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(acos_f16, f16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_bf16, bf16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_f32, f32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_f64, f64, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_bool, bool, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i8, i8, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i16, i16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i32, i32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_i64, i64, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u8, u8, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u16, u16, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u32, u32, FloatOutUnaryPromote, ACos);
DEFINE_UNARY_KERNEL(acos_u64, u64, FloatOutUnaryPromote, ACos);
