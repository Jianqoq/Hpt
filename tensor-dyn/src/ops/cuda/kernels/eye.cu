#include <cuda_fp16.h>

#define DEFINE_KERNEL(func_name, type, one_value)                               \
    extern "C" __global__ void func_name(type *out, size_t rows, size_t cols, int k) \
    {                                                                                \
        size_t row = blockIdx.x * blockDim.x + threadIdx.x;                          \
        if (row < rows)                                                              \
        {                                                                            \
            size_t diag_col = row + k;                                               \
            if (diag_col < cols)                                                     \
            {                                                                        \
                out[row * cols + diag_col] = one_value;                              \
            }                                                                        \
        }                                                                            \
    }

DEFINE_KERNEL(eye_f16, half, __float2half(1.0f));
DEFINE_KERNEL(eye_f32, float, 1.0f);
DEFINE_KERNEL(eye_f64, double, 1.0);

DEFINE_KERNEL(eye_i8, char, 1);
DEFINE_KERNEL(eye_i16, short, 1);
DEFINE_KERNEL(eye_i32, int, 1);
DEFINE_KERNEL(eye_i64, long long, 1);

DEFINE_KERNEL(eye_u8, unsigned char, 1);
DEFINE_KERNEL(eye_u16, unsigned short, 1);
DEFINE_KERNEL(eye_u32, unsigned int, 1);
DEFINE_KERNEL(eye_u64, unsigned long long, 1);