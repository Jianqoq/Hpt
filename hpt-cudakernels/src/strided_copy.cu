#include "utils/fast_divmod.cuh"
#include "utils/index_calculator.cuh"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

template <typename T>
__device__ void strided_copy(T *dst, T *src, FastDivmod *shape, int32_t *strides, int32_t ndim, int64_t size)
{
    UncontiguousIndexCalculator<T> calculator(src, shape, strides, ndim);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        dst[i] = calculator.get(i);
    }
}

#define DECLARE_STRIDED_COPY(TYPE, RUST_TYPE)                                                                                                        \
    extern "C" __global__ void strided_copy_##RUST_TYPE(TYPE *dst, TYPE *src, FastDivmod *shape, int32_t *strides, int32_t ndim, int64_t size) \
    {                                                                                                                                                \
        strided_copy<TYPE>(dst, src, shape, strides, ndim, size);                                                                                    \
    }

DECLARE_STRIDED_COPY(bool, bool)
DECLARE_STRIDED_COPY(int8_t, i8)
DECLARE_STRIDED_COPY(int16_t, i16)
DECLARE_STRIDED_COPY(int32_t, i32)
DECLARE_STRIDED_COPY(int64_t, i64)
DECLARE_STRIDED_COPY(uint8_t, u8)
DECLARE_STRIDED_COPY(uint16_t, u16)
DECLARE_STRIDED_COPY(uint32_t, u32)
DECLARE_STRIDED_COPY(uint64_t, u64)
DECLARE_STRIDED_COPY(float, f32)
DECLARE_STRIDED_COPY(double, f64)
DECLARE_STRIDED_COPY(half, f16)
DECLARE_STRIDED_COPY(__nv_bfloat16, bf16)
