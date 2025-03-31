#include "declare_macros.cuh"
#include "../utils/type_cast.cuh"
#include "../utils/check_type.cuh"
#include "reduce_classes.cuh"

DECLARE_KERNEL(bool, bool, logsumexp, LogSumExp)
DECLARE_KERNEL(u8, u8, logsumexp, LogSumExp)
DECLARE_KERNEL(u16, u16, logsumexp, LogSumExp)
DECLARE_KERNEL(u32, u32, logsumexp, LogSumExp)
DECLARE_KERNEL(u64, u64, logsumexp, LogSumExp)
DECLARE_KERNEL(i8, i8, logsumexp, LogSumExp)
DECLARE_KERNEL(i16, i16, logsumexp, LogSumExp)
DECLARE_KERNEL(i32, i32, logsumexp, LogSumExp)
DECLARE_KERNEL(i64, i64, logsumexp, LogSumExp)
DECLARE_KERNEL(f32, f32, logsumexp, LogSumExp)
DECLARE_KERNEL(f64, f64, logsumexp, LogSumExp)
DECLARE_KERNEL(f16, f16, logsumexp, LogSumExp)
DECLARE_KERNEL(bf16, bf16, logsumexp, LogSumExp)
