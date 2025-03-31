#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/check_type.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(half, bool, reducel2, ReduceL2)
DECLARE_KERNEL(half, u8, reducel2, ReduceL2)
DECLARE_KERNEL(half, u16, reducel2, ReduceL2)
DECLARE_KERNEL(float, u32, reducel2, ReduceL2)
DECLARE_KERNEL(double, u64, reducel2, ReduceL2)
DECLARE_KERNEL(half, i8, reducel2, ReduceL2)
DECLARE_KERNEL(half, i16, reducel2, ReduceL2)
DECLARE_KERNEL(float, i32, reducel2, ReduceL2)
DECLARE_KERNEL(double, i64, reducel2, ReduceL2)
DECLARE_KERNEL(float, f32, reducel2, ReduceL2)
DECLARE_KERNEL(double, f64, reducel2, ReduceL2)
DECLARE_KERNEL(half, f16, reducel2, ReduceL2)
DECLARE_KERNEL(bf16, bf16, reducel2, ReduceL2)
