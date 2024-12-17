#include <cuda_fp16.h>

#define DEFINE_LOGSPACE_KERNEL(func_name, vec_type, type, vec_size)                             \
    extern "C" __global__ void func_name(type *out, type base, type start, type step, size_t n) \
    {                                                                                           \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                     \
        size_t stride = blockDim.x * gridDim.x;                                                 \
        for (size_t i = idx; i < n / vec_size; i += stride)                                     \
        {                                                                                       \
            vec_type##vec_size value;                                                           \
            value.x = pow(base, start + step * (i * vec_size + 0));                             \
            value.y = pow(base, start + step * (i * vec_size + 1));                             \
            value.z = pow(base, start + step * (i * vec_size + 2));                             \
            value.w = pow(base, start + step * (i * vec_size + 3));                             \
            ((vec_type##vec_size *)out)[i] = value;                                             \
        }                                                                                       \
        if (idx == 0)                                                                           \
        {                                                                                       \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                              \
            {                                                                                   \
                type exponent = start + step * i;                                               \
                out[i] = pow(base, exponent);                                                   \
            }                                                                                   \
        }                                                                                       \
    }

#define DEFINE_LOGSPACE_KERNEL_CAST(func_name, vec_type, type, vec_size, cast)                 \
    extern "C" __global__ void func_name(type *out, type base, type start, type step, size_t n) \
    {                                                                                           \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                     \
        size_t stride = blockDim.x * gridDim.x;                                                 \
        for (size_t i = idx; i < n / vec_size; i += stride)                                     \
        {                                                                                       \
            vec_type##vec_size value;                                                           \
            value.x = (type)pow((cast)base, (cast)(start + step * (i * vec_size + 0)));         \
            value.y = (type)pow((cast)base, (cast)(start + step * (i * vec_size + 1)));         \
            value.z = (type)pow((cast)base, (cast)(start + step * (i * vec_size + 2)));         \
            value.w = (type)pow((cast)base, (cast)(start + step * (i * vec_size + 3)));         \
            ((vec_type##vec_size *)out)[i] = value;                                             \
        }                                                                                       \
        if (idx == 0)                                                                           \
        {                                                                                       \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                              \
            {                                                                                   \
                type exponent = start + step * i;                                               \
                out[i] = (type)pow((cast)base, (cast)exponent);                                 \
            }                                                                                   \
        }                                                                                       \
    }

extern "C" __global__ void logspace_f16(half *out, half base, half start, half step, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        __half2 value;
        value.x = pow(base, __half2float(start) + __half2float(step) * (i * 2 + 0));
        value.y = pow(base, __half2float(start) + __half2float(step) * (i * 2 + 1));
        ((__half2 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (size_t i = (n / 2) * 2; i < n; i++)
        {
            float exponent = __half2float(start) + __half2float(step) * i;
            out[i] = pow(base, exponent);
        }
    }
};
DEFINE_LOGSPACE_KERNEL(logspace_f32, float, float, 4);
DEFINE_LOGSPACE_KERNEL(logspace_f64, double, double, 4);

DEFINE_LOGSPACE_KERNEL_CAST(logspace_i8, char, char, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_i16, short, short, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_i32, int, int, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_i64, longlong, long long, 4, double);

DEFINE_LOGSPACE_KERNEL_CAST(logspace_u8, uchar, unsigned char, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_u16, ushort, unsigned short, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_u32, uint, unsigned int, 4, float);
DEFINE_LOGSPACE_KERNEL_CAST(logspace_u64, ulonglong, unsigned long long, 4, double);