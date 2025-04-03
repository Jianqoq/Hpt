#include "declare_macros.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, prod, Prod)
DECLARE_KERNEL(u8, u8, prod, Prod)
DECLARE_KERNEL(u16, u16, prod, Prod)
DECLARE_KERNEL(u32, u32, prod, Prod)
DECLARE_KERNEL(u64, u64, prod, Prod)
DECLARE_KERNEL(i8, i8, prod, Prod)
DECLARE_KERNEL(i16, i16, prod, Prod)
DECLARE_KERNEL(i32, i32, prod, Prod)
DECLARE_KERNEL(i64, i64, prod, Prod)
DECLARE_KERNEL(f32, f32, prod, Prod)
DECLARE_KERNEL(f64, f64, prod, Prod)
DECLARE_KERNEL(f16, f16, prod, Prod)
DECLARE_KERNEL(bf16, bf16, prod, Prod)
