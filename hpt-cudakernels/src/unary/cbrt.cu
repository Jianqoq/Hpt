#include "unary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"
#include "../utils/check_type.cuh"
#include "unary_classes.cuh"

DEFINE_UNARY_KERNEL(cbrt_f16, f16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_bf16, bf16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_f32, f32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_f64, f64, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_bool, bool, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i8, i8, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i16, i16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i32, i32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_i64, i64, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u8, u8, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u16, u16, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u32, u32, FloatOutUnaryPromote, Cbrt);
DEFINE_UNARY_KERNEL(cbrt_u64, u64, FloatOutUnaryPromote, Cbrt);
