#include <cuda_fp16.h>

#define DEFINE_KERNEL(func_name, type, one, zero)                                                \
    extern "C" __global__ void func_name(type *out, size_t rows, size_t cols, int k, bool lower) \
    {                                                                                            \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                      \
        size_t size = rows * cols;                                                               \
        while (idx < size)                                                                       \
        {                                                                                        \
            size_t row = idx / cols;                                                             \
            size_t col = idx % cols;                                                             \
            if (lower)                                                                           \
            {                                                                                    \
                if (col <= row + k)                                                              \
                {                                                                                \
                    out[idx] = one;                                                              \
                }                                                                                \
                else                                                                             \
                {                                                                                \
                    out[idx] = zero;                                                             \
                }                                                                                \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                if (col <= row + k - 1)                                                          \
                {                                                                                \
                    out[idx] = zero;                                                             \
                }                                                                                \
                else                                                                             \
                {                                                                                \
                    out[idx] = one;                                                              \
                }                                                                                \
            }                                                                                    \
            idx += blockDim.x * gridDim.x;                                                       \
        }                                                                                        \
    }

DEFINE_KERNEL(triu_f16, half, __float2half(1.0f), __float2half(0.0f));
DEFINE_KERNEL(triu_f32, float, 1.0f, 0.0f);
DEFINE_KERNEL(triu_f64, double, 1.0, 0.0);

DEFINE_KERNEL(triu_i8, char, 1, 0);
DEFINE_KERNEL(triu_i16, short, 1, 0);
DEFINE_KERNEL(triu_i32, int, 1, 0);
DEFINE_KERNEL(triu_i64, long long, 1, 0);

DEFINE_KERNEL(triu_u8, unsigned char, 1, 0);
DEFINE_KERNEL(triu_u16, unsigned short, 1, 0);
DEFINE_KERNEL(triu_u32, unsigned int, 1, 0);
DEFINE_KERNEL(triu_u64, unsigned long long, 1, 0);