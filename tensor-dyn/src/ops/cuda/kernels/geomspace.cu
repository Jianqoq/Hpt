#include <cuda_fp16.h>

#define DEFINE_GEOMSPACE_KERNEL(func_name, vec_type, type, vec_size, ten)                                               \
    extern "C" __global__ void func_name(type *out, type start, type step, bool neg, size_t n)                          \
    {                                                                                                                   \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                                             \
        size_t stride = blockDim.x * gridDim.x;                                                                         \
        for (size_t i = idx; i < n / vec_size; i += stride)                                                             \
        {                                                                                                               \
            vec_type##vec_size value;                                                                                   \
            value.x = neg ? -pow(ten, start + step * (i * vec_size + 0)) : pow(ten, start + step * (i * vec_size + 0)); \
            value.y = neg ? -pow(ten, start + step * (i * vec_size + 1)) : pow(ten, start + step * (i * vec_size + 1)); \
            value.z = neg ? -pow(ten, start + step * (i * vec_size + 2)) : pow(ten, start + step * (i * vec_size + 2)); \
            value.w = neg ? -pow(ten, start + step * (i * vec_size + 3)) : pow(ten, start + step * (i * vec_size + 3)); \
            ((vec_type##vec_size *)out)[i] = value;                                                                     \
        }                                                                                                               \
        if (idx == 0)                                                                                                   \
        {                                                                                                               \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                                                      \
            {                                                                                                           \
                type exponent = start + step * i;                                                                       \
                out[i] = neg ? -pow(ten, exponent) : pow(ten, exponent);                                                \
            }                                                                                                           \
        }                                                                                                               \
    }

#define DEFINE_GEOMSPACE_KERNEL_CAST(func_name, vec_type, type, vec_size, cast, ten)                                                                \
    extern "C" __global__ void func_name(type *out, type start, type step, bool neg, size_t n)                                                      \
    {                                                                                                                                               \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                         \
        size_t stride = blockDim.x * gridDim.x;                                                                                                     \
        for (size_t i = idx; i < n / vec_size; i += stride)                                                                                         \
        {                                                                                                                                           \
            vec_type##vec_size value;                                                                                                               \
            value.x = neg ? -pow((cast)ten, (cast)(start + step * (i * vec_size + 0))) : pow((cast)ten, (cast)(start + step * (i * vec_size + 0))); \
            value.y = neg ? -pow((cast)ten, (cast)(start + step * (i * vec_size + 1))) : pow((cast)ten, (cast)(start + step * (i * vec_size + 1))); \
            value.z = neg ? -pow((cast)ten, (cast)(start + step * (i * vec_size + 2))) : pow((cast)ten, (cast)(start + step * (i * vec_size + 2))); \
            value.w = neg ? -pow((cast)ten, (cast)(start + step * (i * vec_size + 3))) : pow((cast)ten, (cast)(start + step * (i * vec_size + 3))); \
            ((vec_type##vec_size *)out)[i] = value;                                                                                                 \
        }                                                                                                                                           \
        if (idx == 0)                                                                                                                               \
        {                                                                                                                                           \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                                                                                  \
            {                                                                                                                                       \
                type exponent = start + step * i;                                                                                                   \
                out[i] = neg ? -pow((cast)ten, (cast)exponent) : pow((cast)ten, (cast)exponent);                                                    \
            }                                                                                                                                       \
        }                                                                                                                                           \
    }

extern "C" __global__ void geomspace_f16(half *out, half base, half start, half step, bool neg, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        __half2 value;
        value.x = neg ? -pow(10.0f, __half2float(start) + __half2float(step) * (i * 2 + 0)) : pow(10.0f, __half2float(start) + __half2float(step) * (i * 2 + 0));
        value.y = neg ? -pow(10.0f, __half2float(start) + __half2float(step) * (i * 2 + 1)) : pow(10.0f, __half2float(start) + __half2float(step) * (i * 2 + 1));
        ((__half2 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (size_t i = (n / 2) * 2; i < n; i++)
        {
            float exponent = __half2float(start) + __half2float(step) * i;
            out[i] = neg ? -pow(10.0f, exponent) : pow(10.0f, exponent);
        }
    }
};

DEFINE_GEOMSPACE_KERNEL(geomspace_f32, float, float, 4, 10.0f);
DEFINE_GEOMSPACE_KERNEL(geomspace_f64, double, double, 4, 10.0);

DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_i8, char, char, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_i16, short, short, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_i32, int, int, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_i64, longlong, long long, 4, double, 10.0);

DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_u8, uchar, unsigned char, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_u16, ushort, unsigned short, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_u32, uint, unsigned int, 4, float, 10.0f);
DEFINE_GEOMSPACE_KERNEL_CAST(geomspace_u64, ulonglong, unsigned long long, 4, double, 10.0);