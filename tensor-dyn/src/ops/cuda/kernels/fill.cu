#include <cuda_fp16.h>

#define MAKE_VEC4(vec_type, vec_size, value) make_##vec_type##vec_size(value, value, value, value)
#define MAKE_VEC3(vec_type, vec_size, value) make_##vec_type##vec_size(value, value, value)
#define MAKE_VEC2(vec_type, vec_size, value) make_##vec_type##vec_size(value, value)

#define DEFINE_FILL_KERNEL(func_name, vec_type, type, vec_size)                       \
    extern "C" __global__ void func_name(type *out, type value, size_t N)                        \
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

extern "C" __global__ void fill_f16_vec2(half *out, half value, size_t N)
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
DEFINE_FILL_KERNEL(fill_f32_vec4, float, float, 4);
DEFINE_FILL_KERNEL(fill_f32_vec3, float, float, 3);
DEFINE_FILL_KERNEL(fill_f32_vec2, float, float, 2);
DEFINE_FILL_KERNEL(fill_f64_vec4, double, double, 4);
DEFINE_FILL_KERNEL(fill_f64_vec3, double, double, 3);
DEFINE_FILL_KERNEL(fill_f64_vec2, double, double, 2);

DEFINE_FILL_KERNEL(fill_i8_vec4, char, char, 4);
DEFINE_FILL_KERNEL(fill_i8_vec3, char, char, 3);
DEFINE_FILL_KERNEL(fill_i8_vec2, char, char, 2);
DEFINE_FILL_KERNEL(fill_i16_vec4, short, short, 4);
DEFINE_FILL_KERNEL(fill_i16_vec3, short, short, 3);
DEFINE_FILL_KERNEL(fill_i16_vec2, short, short, 2);
DEFINE_FILL_KERNEL(fill_i32_vec4, int, int, 4);
DEFINE_FILL_KERNEL(fill_i32_vec3, int, int, 3);
DEFINE_FILL_KERNEL(fill_i32_vec2, int, int, 2);
DEFINE_FILL_KERNEL(fill_i64_vec4, longlong, long long, 4);
DEFINE_FILL_KERNEL(fill_i64_vec3, longlong, long long, 3);
DEFINE_FILL_KERNEL(fill_i64_vec2, longlong, long long, 2);

DEFINE_FILL_KERNEL(fill_u8_vec4, uchar, unsigned char, 4);
DEFINE_FILL_KERNEL(fill_u8_vec3, uchar, unsigned char, 3);
DEFINE_FILL_KERNEL(fill_u8_vec2, uchar, unsigned char, 2);
DEFINE_FILL_KERNEL(fill_u16_vec4, ushort, unsigned short, 4);
DEFINE_FILL_KERNEL(fill_u16_vec3, ushort, unsigned short, 3);
DEFINE_FILL_KERNEL(fill_u16_vec2, ushort, unsigned short, 2);
DEFINE_FILL_KERNEL(fill_u32_vec4, uint, unsigned int, 4);
DEFINE_FILL_KERNEL(fill_u32_vec3, uint, unsigned int, 3);
DEFINE_FILL_KERNEL(fill_u32_vec2, uint, unsigned int, 2);
DEFINE_FILL_KERNEL(fill_u64_vec4, ulonglong, unsigned long long, 4);
DEFINE_FILL_KERNEL(fill_u64_vec3, ulonglong, unsigned long long, 3);
DEFINE_FILL_KERNEL(fill_u64_vec2, ulonglong, unsigned long long, 2);
