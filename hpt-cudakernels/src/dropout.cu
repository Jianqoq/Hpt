#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include "utils/type_alias.cuh"

#define DEFINE_DROPOUT_KERNEL(rust_type)                                                                                                                            \
    extern "C" __global__ void dropout_##rust_type(rust_type *out, const rust_type *input, float prob, rust_type scale, const unsigned long long seed, size_t size) \
    {                                                                                                                                                               \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                            \
        if (idx < size)                                                                                                                                             \
        {                                                                                                                                                           \
            curandState state;                                                                                                                                      \
            curand_init(seed, idx, 0, &state);                                                                                                                      \
            float rand = curand_uniform(&state);                                                                                                                    \
            out[idx] = rand < prob ? 0 : input[idx] * scale;                                                                                                        \
        }                                                                                                                                                           \
    }                                                                                                                                                               \
    extern "C" __global__ void dropout_uncontiguous_##rust_type(                                                                                                    \
        rust_type *out,                                                                                                                                             \
        const rust_type *input,                                                                                                                                     \
        float prob,                                                                                                                                                 \
        rust_type scale,                                                                                                                                            \
        const unsigned long long seed,                                                                                                                              \
        const long long *shape,                                                                                                                                     \
        const long long *strides,                                                                                                                                   \
        size_t ndim,                                                                                                                                                \
        size_t size)                                                                                                                                                \
    {                                                                                                                                                               \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                            \
        if (idx < size)                                                                                                                                             \
        {                                                                                                                                                           \
            long long a_amount = idx;                                                                                                                               \
            long long a_offset = 0;                                                                                                                                 \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                     \
            {                                                                                                                                                       \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                     \
                a_amount /= shape[j];                                                                                                                               \
            }                                                                                                                                                       \
            curandState state;                                                                                                                                      \
            curand_init(seed, idx, 0, &state);                                                                                                                      \
            float rand = curand_uniform(&state);                                                                                                                    \
            out[idx] = rand < prob ? 0 : input[a_offset] * scale;                                                                                                   \
        }                                                                                                                                                           \
    }

DEFINE_DROPOUT_KERNEL(bool)
DEFINE_DROPOUT_KERNEL(i8)
DEFINE_DROPOUT_KERNEL(i16)
DEFINE_DROPOUT_KERNEL(i32)
DEFINE_DROPOUT_KERNEL(i64)
DEFINE_DROPOUT_KERNEL(u8)
DEFINE_DROPOUT_KERNEL(u16)
DEFINE_DROPOUT_KERNEL(u32)
DEFINE_DROPOUT_KERNEL(u64)
DEFINE_DROPOUT_KERNEL(f32)
DEFINE_DROPOUT_KERNEL(f64)
DEFINE_DROPOUT_KERNEL(f16)
DEFINE_DROPOUT_KERNEL(bf16)
