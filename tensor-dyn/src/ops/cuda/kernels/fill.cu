#include <cuda_fp16.h>

#define MAKE_VEC4(vec_type, vec_size, value) make_##vec_type##vec_size(value, value, value, value)
#define MAKE_VEC3(vec_type, vec_size, value) make_##vec_type##vec_size(value, value, value)
#define MAKE_VEC2(vec_type, vec_size, value) make_##vec_type##vec_size(value, value)

#define DEFINE_KERNEL(func_name, vec_type, type, vec_size)                            \
    extern "C" __global__ void func_name(type *out, type value, size_t N)             \
    {                                                                                 \
        vec_type##vec_size *out_vec = (vec_type##vec_size *)out;                      \
        vec_type##vec_size value_vec = MAKE_VEC##vec_size(vec_type, vec_size, value); \
                                                                                      \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
        size_t stride = blockDim.x * gridDim.x;                                       \
        size_t N_vec = N / vec_size;                                                  \
                                                                                      \
        for (size_t i = idx; i < N_vec; i += stride)                                  \
        {                                                                             \
            out_vec[i] = value_vec;                                                   \
        }                                                                             \
                                                                                      \
        if (idx == 0)                                                                 \
        {                                                                             \
            for (size_t i = N_vec * vec_size; i < N; i++)                             \
            {                                                                         \
                out[i] = value;                                                       \
            }                                                                         \
        }                                                                             \
    }

extern "C" __global__ void fill_f16(half *out, half value, size_t N)
{
    __half2 *out_vec = (__half2 *)out;
    __half2 value_vec = make_half2(value, value);
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t N_vec = N / 2;
    for (size_t i = idx; i < N_vec; i += stride)
    {
        out_vec[i] = value_vec;
    }
    if (idx == 0)
    {
        for (size_t i = N_vec * 2; i < N; i++)
        {
            out[i] = value;
        }
    }
};
DEFINE_KERNEL(fill_f32, float, float, 4);
DEFINE_KERNEL(fill_f64, double, double, 4);

DEFINE_KERNEL(fill_i8, char, char, 4);
DEFINE_KERNEL(fill_i16, short, short, 4);
DEFINE_KERNEL(fill_i32, int, int, 4);
DEFINE_KERNEL(fill_i64, longlong, long long, 4);
DEFINE_KERNEL(fill_u8, uchar, unsigned char, 4);
DEFINE_KERNEL(fill_u16, ushort, unsigned short, 4);
DEFINE_KERNEL(fill_u32, uint, unsigned int, 4);
DEFINE_KERNEL(fill_u64, ulonglong, unsigned long long, 4);
