#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
DEFINE_KERNEL(fill_f16, half, half, 2);
extern "C" __global__ void fill_bf16(__nv_bfloat16 *out, __nv_bfloat16 value, size_t N)
{
    __nv_bfloat162 *out_vec = (__nv_bfloat162 *)out;
    __nv_bfloat162 value_vec = make_bfloat162(value, value);
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
extern "C" __global__ void fill_bool(bool *out, bool value, size_t N)
{
    bool *out_vec = (bool *)out;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride)
    {
        out_vec[i] = value;
    }
};

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
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        value.x = neg ? -powf(10.0f, start + step * __ull2half_rn((i * 2 + 0))) : powf(10.0f, start + step * __ull2half_rn((i * 2 + 0)));
        value.y = neg ? -powf(10.0f, start + step * __ull2half_rn((i * 2 + 1))) : powf(10.0f, start + step * __ull2half_rn((i * 2 + 1)));
#else
        value.x = neg ? -powf(10.0f, start + step * __uint2half_rn((i * 2 + 0))) : powf(10.0f, start + step * __uint2half_rn((i * 2 + 0)));
        value.y = neg ? -powf(10.0f, start + step * __uint2half_rn((i * 2 + 1))) : powf(10.0f, start + step * __uint2half_rn((i * 2 + 1)));
#endif
        ((half2 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (size_t i = (n / 2) * 2; i < n; i++)
        {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
            float exponent = __ull2float_rn(start + step * __ull2half_rn(i));
#else
            float exponent = __uint2float_rn(start + step * __uint2half_rn(i));
#endif
            out[i] = neg ? -pow(10.0f, exponent) : pow(10.0f, exponent);
        }
    }
};

extern "C" __global__ void geomspace_bf16(__nv_bfloat16 *out, __nv_bfloat16 base, __nv_bfloat16 start, __nv_bfloat16 step, bool neg, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        __nv_bfloat162 value;
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        value.x = neg ? -powf(10.0f, start + step * __ull2bfloat16_rn((i * 2 + 0))) : powf(10.0f, start + step * __ull2bfloat16_rn((i * 2 + 0)));
        value.y = neg ? -powf(10.0f, start + step * __ull2bfloat16_rn((i * 2 + 1))) : powf(10.0f, start + step * __ull2bfloat16_rn((i * 2 + 1)));
#else
        value.x = neg ? -powf(10.0f, start + step * __uint2bfloat16_rn((i * 2 + 0))) : powf(10.0f, start + step * __uint2bfloat16_rn((i * 2 + 0)));
        value.y = neg ? -powf(10.0f, start + step * __uint2bfloat16_rn((i * 2 + 1))) : powf(10.0f, start + step * __uint2bfloat16_rn((i * 2 + 1)));
#endif
        ((__nv_bfloat162 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (size_t i = (n / 2) * 2; i < n; i++)
        {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
            float exponent = __ull2float_rn(start + step * __ull2bfloat16_rn(i));
#else
            float exponent = __uint2float_rn(start + step * __uint2bfloat16_rn(i));
#endif
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

#define DEFINE_LINSPACE_KERNEL(func_name, vec_type, type, vec_size)                                              \
    extern "C" __global__ void func_name(type *out, type start, type step, type end, bool include_end, size_t n) \
    {                                                                                                            \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                                                      \
        size_t stride = blockDim.x * gridDim.x;                                                                  \
                                                                                                                 \
        for (size_t i = idx; i < n / vec_size; i += stride)                                                      \
        {                                                                                                        \
            ##vec_type##vec_size value;                                                                          \
            size_t base = i * 4;                                                                                 \
            value.x = start + step * (i * vec_size + 0);                                                         \
            value.y = start + step * (i * vec_size + 1);                                                         \
            value.z = start + step * (i * vec_size + 2);                                                         \
            value.w = start + step * (i * vec_size + 3);                                                         \
            if (include_end)                                                                                     \
            {                                                                                                    \
                if (base + 3 >= n - 1)                                                                           \
                {                                                                                                \
                    if (base + 3 == n - 1)                                                                       \
                        value.w = end;                                                                           \
                    if (base + 2 == n - 1)                                                                       \
                        value.z = end;                                                                           \
                    if (base + 1 == n - 1)                                                                       \
                        value.y = end;                                                                           \
                    if (base == n - 1)                                                                           \
                        value.x = end;                                                                           \
                }                                                                                                \
            }                                                                                                    \
            ((##vec_type##vec_size *)out)[i] = value;                                                            \
        }                                                                                                        \
                                                                                                                 \
        if (idx == 0)                                                                                            \
        {                                                                                                        \
            for (size_t i = (n / vec_size) * vec_size; i < n; i++)                                               \
            {                                                                                                    \
                if (include_end && i == n - 1)                                                                   \
                {                                                                                                \
                    out[i] = end;                                                                                \
                }                                                                                                \
                else                                                                                             \
                {                                                                                                \
                    out[i] = start + step * i;                                                                   \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }
extern "C" __global__ void linspace_f16(half *out, half start, half step, half end, bool include_end, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        half2 value;
        size_t base = i * 2;
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        value.x = start + step * __ull2half_rn((base + 0));
        value.y = start + step * __ull2half_rn((base + 1));
#else
        value.x = start + step * __uint2half_rn((base + 0));
        value.y = start + step * __uint2half_rn((base + 1));
#endif
        if (include_end)
        {
            if (base + 1 >= n - 1)
            {
                if (base + 1 == n - 1)
                    value.y = end;
                if (base == n - 1)
                    value.x = end;
            }
        }
        ((half2 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (long long i = (n / 2) * 2; i < n; i++)
        {
            if (include_end && i == n - 1)
            {
                out[i] = end;
            }
            else
            {
                out[i] = start + step * __ll2half_rn(i);
            }
        }
    }
};
extern "C" __global__ void linspace_bf16(__nv_bfloat16 *out, __nv_bfloat16 start, __nv_bfloat16 step, __nv_bfloat16 end, bool include_end, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n / 2; i += stride)
    {
        __nv_bfloat162 value;
        size_t base = i * 2;
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        value.x = start + step * __ull2bfloat16_rn((base + 0));
        value.y = start + step * __ull2bfloat16_rn((base + 1));
#else
        value.x = start + step * __uint2bfloat16_rn((base + 0));
        value.y = start + step * __uint2bfloat16_rn((base + 1));
#endif
        if (include_end)
        {
            if (base + 1 >= n - 1)
            {
                if (base + 1 == n - 1)
                    value.y = end;
                if (base == n - 1)
                    value.x = end;
            }
        }
        ((__nv_bfloat162 *)out)[i] = value;
    }
    if (idx == 0)
    {
        for (long long i = (n / 2) * 2; i < n; i++)
        {
            if (include_end && i == n - 1)
            {
                out[i] = end;
            }
            else
            {
                out[i] = start + step * __ll2bfloat16_rn(i);
            }
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

#define DEFINE_TRIU_KERNEL(func_name, type, one, zero)                                           \
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

DEFINE_TRIU_KERNEL(triu_bool, bool, true, false);
DEFINE_TRIU_KERNEL(triu_f16, half, __float2half(1.0f), __float2half(0.0f));
DEFINE_TRIU_KERNEL(triu_bf16, __nv_bfloat16, __float2bfloat16(1.0f), __float2bfloat16(0.0f));
DEFINE_TRIU_KERNEL(triu_f32, float, 1.0f, 0.0f);
DEFINE_TRIU_KERNEL(triu_f64, double, 1.0, 0.0);

DEFINE_TRIU_KERNEL(triu_i8, char, 1, 0);
DEFINE_TRIU_KERNEL(triu_i16, short, 1, 0);
DEFINE_TRIU_KERNEL(triu_i32, int, 1, 0);
DEFINE_TRIU_KERNEL(triu_i64, long long, 1, 0);

DEFINE_TRIU_KERNEL(triu_u8, unsigned char, 1, 0);
DEFINE_TRIU_KERNEL(triu_u16, unsigned short, 1, 0);
DEFINE_TRIU_KERNEL(triu_u32, unsigned int, 1, 0);
DEFINE_TRIU_KERNEL(triu_u64, unsigned long long, 1, 0);

#define ARANGE_MAKE_VEC4(vec_type, vec_size, value) make_##vec_type##vec_size(value, value + step, value + step * 2, value + step * 3)
#define ARANGE_MAKE_VEC3(vec_type, vec_size, value) make_##vec_type##vec_size(value, value + step, value + step * 2)
#define ARANGE_MAKE_VEC2(vec_type, vec_size, value) make_##vec_type##vec_size(value, value + step)

#define DEFINE_ARANGE_KERNEL(func_name, vec_type, type, vec_size)                    \
    extern "C" __global__ void func_name(type *out, type start, type step, size_t N) \
    {                                                                                \
        vec_type##vec_size *out_vec = (vec_type##vec_size *)out;                     \
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;                          \
        size_t stride = blockDim.x * gridDim.x;                                      \
        size_t N_vec = N / vec_size;                                                 \
                                                                                     \
        for (size_t i = idx; i < N_vec; i += stride)                                 \
        {                                                                            \
            type base = start + (i * vec_size) * step;                               \
            vec_type##vec_size value_vec = ARANGE_MAKE_VEC##vec_size(                \
                vec_type, vec_size, base);                                           \
            out_vec[i] = value_vec;                                                  \
        }                                                                            \
                                                                                     \
        if (idx == 0)                                                                \
        {                                                                            \
            for (size_t i = N_vec * vec_size; i < N; i++)                            \
            {                                                                        \
                out[i] = start + i * step;                                           \
            }                                                                        \
        }                                                                            \
    }

extern "C" __global__ void arange_f16(half *out, half start, half step, size_t N)
{
    half2 *out_vec = (half2 *)out;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t N_vec = N / 2;
    for (size_t i = idx; i < N_vec; i += stride)
    {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        half base = start + step * __ull2half_rn((i * 2));
#else
        half base = start + step * __uint2half_rn((i * 2));
#endif
        half2 value_vec = make_half2(base, base + step);
        out_vec[i] = value_vec;
    }
    if (idx == 0)
    {
        for (size_t i = N_vec * 2; i < N; i++)
        {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
            out[i] = start + __ull2half_rn(i) * step;
#else
            out[i] = start + __uint2half_rn(i) * step;
#endif
        }
    }
};

extern "C" __global__ void arange_bf16(__nv_bfloat16 *out, __nv_bfloat16 start, __nv_bfloat16 step, size_t N)
{
    __nv_bfloat162 *out_vec = (__nv_bfloat162 *)out;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t N_vec = N / 2;
    for (size_t i = idx; i < N_vec; i += stride)
    {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
        __nv_bfloat16 base = start + step * __ull2bfloat16_rn((i * 2));
#else
        __nv_bfloat16 base = start + step * __uint2bfloat16_rn((i * 2));
#endif
        __nv_bfloat162 value_vec = make_bfloat162(base, base + step);
        out_vec[i] = value_vec;
    }
    if (idx == 0)
    {
        for (size_t i = N_vec * 2; i < N; i++)
        {
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__)
            out[i] = start + __ull2bfloat16_rn(i) * step;
#else
            out[i] = start + __uint2bfloat16_rn(i) * step;
#endif
        }
    }
};

extern "C" __global__ void arange_bool(bool *out, bool start, bool step, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < N; i += stride)
    {
        if (step)
        {
            out[i] = start ^ (i & 1);
        }
        else
        {
            out[i] = start;
        }
    }
};

DEFINE_ARANGE_KERNEL(arange_f32, float, float, 4);
DEFINE_ARANGE_KERNEL(arange_f64, double, double, 4);

DEFINE_ARANGE_KERNEL(arange_i8, char, char, 4);
DEFINE_ARANGE_KERNEL(arange_i16, short, short, 4);
DEFINE_ARANGE_KERNEL(arange_i32, int, int, 4);
DEFINE_ARANGE_KERNEL(arange_i64, longlong, long long, 4);

DEFINE_ARANGE_KERNEL(arange_u8, uchar, unsigned char, 4);
DEFINE_ARANGE_KERNEL(arange_u16, ushort, unsigned short, 4);
DEFINE_ARANGE_KERNEL(arange_u32, uint, unsigned int, 4);
DEFINE_ARANGE_KERNEL(arange_u64, ulonglong, unsigned long long, 4);