#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(hard_swish_f16, f16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_bf16, bf16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_f32, f32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_f64, f64, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_bool, bool, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i8, i8, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i16, i16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i32, i32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_i64, i64, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u8, u8, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u16, u16, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u32, u32, FloatOutUnaryPromote, HardSwish);
DEFINE_UNARY_KERNEL(hard_swish_u64, u64, FloatOutUnaryPromote, HardSwish);
