#include "declare_macros.cuh"
#include "../utils/type_utils.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, min, Min)
DECLARE_KERNEL(u8, u8, min, Min)
DECLARE_KERNEL(u16, u16, min, Min)
DECLARE_KERNEL(u32, u32, min, Min)
DECLARE_KERNEL(u64, u64, min, Min)
DECLARE_KERNEL(i8, i8, min, Min)
DECLARE_KERNEL(i16, i16, min, Min)
DECLARE_KERNEL(i32, i32, min, Min)
DECLARE_KERNEL(i64, i64, min, Min)
DECLARE_KERNEL(f32, f32, min, Min)
DECLARE_KERNEL(f64, f64, min, Min)
DECLARE_KERNEL(f16, f16, min, Min)
DECLARE_KERNEL(bf16, bf16, min, Min)
