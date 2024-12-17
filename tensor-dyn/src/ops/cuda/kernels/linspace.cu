#include <cuda_fp16.h>

#define DEFINE_LINSPACE_KERNEL(func_name, vec_type, type, vec_size)                  \
    extern "C" __global__ void func_name(type *out, type start, type step, size_t n) \
    {                                                                                \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
        size_t stride = blockDim.x * gridDim.x;                                      \
                                                                                     \
        for (size_t i = idx; i < n / vec_size; i += stride)                          \
        {                                                                            \
            ##vec_type##vec_size value;                                              \
            value.x = start + step * (i * vec_size + 0);                             \
            value.y = start + step * (i * vec_size + 1);                             \
            value.z = start + step * (i * vec_size + 2);                             \
            value.w = start + step * (i * vec_size + 3);                             \
            ((##vec_type##vec_size *)out)[i] = value;                                \
        }                                                                            \
                                                                                     \
        if (idx == 0)                                                                \
        {                                                                            \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                   \
            {                                                                        \
                out[i] = start + step * i;                                           \
            }                                                                        \
        }                                                                            \
    }
extern "C" __global__ void linspace_f16(half *out, float start, float step, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        half2 value;
        value.x = start + step * (i * 2 + 0);
        value.y = start + step * (i * 2 + 1);
        ((half2 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (size_t i = (n / 2) * 2; i < n; i++)
        {
            out[i] = start + step * i;
        }
    }
};
DEFINE_LINSPACE_KERNEL(linspace_f32, float, float, 4);
DEFINE_LINSPACE_KERNEL(linspace_f64, double, double, 4);

DEFINE_LINSPACE_KERNEL(linspace_i8, char, char, 4);
DEFINE_LINSPACE_KERNEL(linspace_i16, short, short, 4);
DEFINE_LINSPACE_KERNEL(linspace_i32, int, int, 4);
DEFINE_LINSPACE_KERNEL(linspace_i64, longlong, long long, 4);

DEFINE_LINSPACE_KERNEL(linspace_u8, uchar, unsigned char, 4);
DEFINE_LINSPACE_KERNEL(linspace_u16, ushort, unsigned short, 4);
DEFINE_LINSPACE_KERNEL(linspace_u32, uint, unsigned int, 4);
DEFINE_LINSPACE_KERNEL(linspace_u64, ulonglong, unsigned long long, 4);