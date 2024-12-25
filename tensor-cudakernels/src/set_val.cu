#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define SET_VAL(rust_type, T)                                                  \
    extern "C" __global__ void set_val_##rust_type(T *dst, T val, size_t size) \
    {                                                                          \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                       \
        if (idx < size)                                                        \
        {                                                                      \
            dst[idx] = val;                                                    \
        }                                                                      \
    }

SET_VAL(f32, float);
SET_VAL(f64, double);
SET_VAL(i32, int);
SET_VAL(i64, long long);
SET_VAL(i16, short);
SET_VAL(i8, char);
SET_VAL(u32, unsigned int);
SET_VAL(u64, unsigned long long);
SET_VAL(u16, unsigned short);
SET_VAL(u8, unsigned char);
SET_VAL(bool, bool);
SET_VAL(bfloat16, __nv_bfloat16);
SET_VAL(half, __nv_half);
