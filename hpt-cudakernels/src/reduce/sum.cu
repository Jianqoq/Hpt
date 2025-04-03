#include "declare_macros.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, sum, Sum)
DECLARE_KERNEL(u8, u8, sum, Sum)
DECLARE_KERNEL(u16, u16, sum, Sum)
DECLARE_KERNEL(u32, u32, sum, Sum)
DECLARE_KERNEL(u64, u64, sum, Sum)
DECLARE_KERNEL(i8, i8, sum, Sum)
DECLARE_KERNEL(i16, i16, sum, Sum)
DECLARE_KERNEL(i32, i32, sum, Sum)
DECLARE_KERNEL(i64, i64, sum, Sum)
DECLARE_KERNEL(f32, f32, sum, Sum)
DECLARE_KERNEL(f64, f64, sum, Sum)
DECLARE_KERNEL(f16, f16, sum, Sum)
DECLARE_KERNEL(bf16, bf16, sum, Sum)
