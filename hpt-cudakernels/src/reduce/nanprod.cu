#include "declare_macros.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, nanprod, NanProd)
DECLARE_KERNEL(u8, u8, nanprod, NanProd)
DECLARE_KERNEL(u16, u16, nanprod, NanProd)
DECLARE_KERNEL(u32, u32, nanprod, NanProd)
DECLARE_KERNEL(u64, u64, nanprod, NanProd)
DECLARE_KERNEL(i8, i8, nanprod, NanProd)
DECLARE_KERNEL(i16, i16, nanprod, NanProd)
DECLARE_KERNEL(i32, i32, nanprod, NanProd)
DECLARE_KERNEL(i64, i64, nanprod, NanProd)
DECLARE_KERNEL(f32, f32, nanprod, NanProd)
DECLARE_KERNEL(f64, f64, nanprod, NanProd)
DECLARE_KERNEL(f16, f16, nanprod, NanProd)
DECLARE_KERNEL(bf16, bf16, nanprod, NanProd)
