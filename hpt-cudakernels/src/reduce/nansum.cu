#include "declare_macros.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, nansum, NanSum)
DECLARE_KERNEL(u8, u8, nansum, NanSum)
DECLARE_KERNEL(u16, u16, nansum, NanSum)
DECLARE_KERNEL(u32, u32, nansum, NanSum)
DECLARE_KERNEL(u64, u64, nansum, NanSum)
DECLARE_KERNEL(i8, i8, nansum, NanSum)
DECLARE_KERNEL(i16, i16, nansum, NanSum)
DECLARE_KERNEL(i32, i32, nansum, NanSum)
DECLARE_KERNEL(i64, i64, nansum, NanSum)
DECLARE_KERNEL(f32, f32, nansum, NanSum)
DECLARE_KERNEL(f64, f64, nansum, NanSum)
DECLARE_KERNEL(f16, f16, nansum, NanSum)
DECLARE_KERNEL(bf16, bf16, nansum, NanSum)
