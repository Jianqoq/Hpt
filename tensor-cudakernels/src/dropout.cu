#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

#define DEFINE_DROPOUT_KERNEL(rust_type, type)                                                                                                       \
    extern "C" __global__ void dropout_##rust_type(type *out, const type *input, float prob, type scale, const unsigned long long seed, size_t size) \
    {                                                                                                                                                \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                             \
        if (idx < size)                                                                                                                              \
        {                                                                                                                                            \
            curandState state;                                                                                                                       \
            curand_init(seed, idx, 0, &state);                                                                                                       \
            float rand = curand_uniform(&state);                                                                                                     \
            out[idx] = rand < prob ? 0 : input[idx] * scale;                                                                                         \
        }                                                                                                                                            \
    }                                                                                                                                                \
    extern "C" __global__ void dropout_uncontiguous_##rust_type(                                                                                     \
        type *out,                                                                                                                                   \
        const type *input,                                                                                                                           \
        float prob,                                                                                                                                  \
        type scale,                                                                                                                                  \
        const unsigned long long seed,                                                                                                               \
        const long long *shape,                                                                                                                      \
        const long long *strides,                                                                                                                    \
        size_t ndim,                                                                                                                                 \
        size_t size)                                                                                                                                 \
    {                                                                                                                                                \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                             \
        if (idx < size)                                                                                                                              \
        {                                                                                                                                            \
            long long a_amount = idx;                                                                                                                \
            long long a_offset = 0;                                                                                                                  \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                      \
            {                                                                                                                                        \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                      \
                a_amount /= shape[j];                                                                                                                \
            }                                                                                                                                        \
            curandState state;                                                                                                                       \
            curand_init(seed, idx, 0, &state);                                                                                                       \
            float rand = curand_uniform(&state);                                                                                                     \
            out[idx] = rand < prob ? 0 : input[a_offset] * scale;                                                                                    \
        }                                                                                                                                            \
    }

DEFINE_DROPOUT_KERNEL(bool, bool)
DEFINE_DROPOUT_KERNEL(i8, char)
DEFINE_DROPOUT_KERNEL(i16, short)
DEFINE_DROPOUT_KERNEL(i32, int)
DEFINE_DROPOUT_KERNEL(i64, long long)
DEFINE_DROPOUT_KERNEL(u8, unsigned char)
DEFINE_DROPOUT_KERNEL(u16, unsigned short)
DEFINE_DROPOUT_KERNEL(u32, unsigned int)
DEFINE_DROPOUT_KERNEL(u64, unsigned long long)
DEFINE_DROPOUT_KERNEL(f32, float)
DEFINE_DROPOUT_KERNEL(f64, double)
DEFINE_DROPOUT_KERNEL(f16, __half)
DEFINE_DROPOUT_KERNEL(bf16, __nv_bfloat16)
