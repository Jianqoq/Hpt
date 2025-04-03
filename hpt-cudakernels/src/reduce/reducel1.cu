#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, reducel1, ReduceL1)
DECLARE_KERNEL(u8, u8, reducel1, ReduceL1)
DECLARE_KERNEL(u16, u16, reducel1, ReduceL1)
DECLARE_KERNEL(u32, u32, reducel1, ReduceL1)
DECLARE_KERNEL(u64, u64, reducel1, ReduceL1)
DECLARE_KERNEL(i8, i8, reducel1, ReduceL1)
DECLARE_KERNEL(i16, i16, reducel1, ReduceL1)
DECLARE_KERNEL(i32, i32, reducel1, ReduceL1)
DECLARE_KERNEL(i64, i64, reducel1, ReduceL1)
DECLARE_KERNEL(f32, f32, reducel1, ReduceL1)
DECLARE_KERNEL(f64, f64, reducel1, ReduceL1)
DECLARE_KERNEL(f16, f16, reducel1, ReduceL1)
DECLARE_KERNEL(bf16, bf16, reducel1, ReduceL1)
