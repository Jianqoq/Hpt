#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/check_type.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, mean, Mean)
DECLARE_KERNEL(u8, u8, mean, Mean)
DECLARE_KERNEL(u16, u16, mean, Mean)
DECLARE_KERNEL(u32, u32, mean, Mean)
DECLARE_KERNEL(u64, u64, mean, Mean)
DECLARE_KERNEL(i8, i8, mean, Mean)
DECLARE_KERNEL(i16, i16, mean, Mean)
DECLARE_KERNEL(i32, i32, mean, Mean)
DECLARE_KERNEL(i64, i64, mean, Mean)
DECLARE_KERNEL(f32, f32, mean, Mean)
DECLARE_KERNEL(f64, f64, mean, Mean)
DECLARE_KERNEL(f16, f16, mean, Mean)
DECLARE_KERNEL(bf16, bf16, mean, Mean)
