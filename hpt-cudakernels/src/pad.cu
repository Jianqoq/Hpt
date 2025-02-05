#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DEFINE_PAD(rust_type, type)                                             \
    extern "C" __global__ void pad_##rust_type(type *out,                       \
                                               type *in,                        \
                                               type pad_value,                  \
                                               const long long *res_shape,      \
                                               const long long *res_strides,    \
                                               const long long *shape,          \
                                               const long long *strides,        \
                                               size_t ndim,                     \
                                               const long long *pads_start,     \
                                               const long long *pads_end,       \
                                               size_t size)                     \
    {                                                                           \
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;         \
        if (idx >= size)                                                        \
            return;                                                             \
        bool is_padding = false;                                                \
        long long amount = idx;                                                 \
        unsigned int coord = 0;                                                 \
        long long inp_idx = 0;                                                  \
        for (int j = ndim - 1; j >= 0; j--)                                     \
        {                                                                       \
            coord = amount % res_shape[j];                                      \
            amount /= res_shape[j];                                             \
            if (coord < pads_start[j] || coord >= (res_shape[j] - pads_end[j])) \
            {                                                                   \
                is_padding = true;                                              \
            }                                                                   \
            inp_idx += (coord - pads_start[j]) * strides[j];                    \
        }                                                                       \
        out[idx] = is_padding ? pad_value : in[inp_idx];                        \
    }

DEFINE_PAD(bool, bool);
DEFINE_PAD(u8, unsigned char);
DEFINE_PAD(i8, char);
DEFINE_PAD(u16, unsigned short);
DEFINE_PAD(i16, short);
DEFINE_PAD(u32, unsigned int);
DEFINE_PAD(i32, int);
DEFINE_PAD(u64, unsigned long long);
DEFINE_PAD(i64, long long);
DEFINE_PAD(f32, float);
DEFINE_PAD(f64, double);
DEFINE_PAD(f16, half);
DEFINE_PAD(bf16, __nv_bfloat16);
