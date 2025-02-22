#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <limits.h>
#include <stdio.h>

#define WRAP 32

#define BOOL_MAX 1
#define I8_MAX SCHAR_MAX
#define I16_MAX SHRT_MAX
#define I32_MAX INT_MAX
#define I64_MAX LLONG_MAX
#define F32_MAX INFINITY
#define F64_MAX INFINITY

#define U8_MAX UCHAR_MAX
#define U16_MAX USHRT_MAX
#define U32_MAX UINT_MAX
#define U64_MAX ULONG_MAX
#define F16_MAX __half((unsigned short)31744)
#define BF16_MAX __nv_bfloat16((unsigned short)0x7F80)

#define gt_bool(a, b) ((bool)((unsigned char)a) > ((unsigned char)b))
#define gt_i8(a, b) ((a) > (b))
#define gt_i16(a, b) ((a) > (b))
#define gt_i32(a, b) ((a) > (b))
#define gt_i64(a, b) ((a) > (b))
#define gt_u8(a, b) ((a) > (b))
#define gt_u16(a, b) ((a) > (b))
#define gt_u32(a, b) ((a) > (b))
#define gt_u64(a, b) ((a) > (b))
#define gt_f32(a, b) ((a) > (b))
#define gt_f64(a, b) ((a) > (b))
#define gt_f16(a, b) (__hgt((a), (b)))
#define gt_bf16(a, b) (__hgt((a), (b)))

#define eq_bool(a, b) ((bool)((unsigned char)a) == ((unsigned char)b))
#define eq_i8(a, b) ((a) == (b))
#define eq_i16(a, b) ((a) == (b))
#define eq_i32(a, b) ((a) == (b))
#define eq_i64(a, b) ((a) == (b))
#define eq_u8(a, b) ((a) == (b))
#define eq_u16(a, b) ((a) == (b))
#define eq_u32(a, b) ((a) == (b))
#define eq_u64(a, b) ((a) == (b))
#define eq_f32(a, b) ((a) == (b))
#define eq_f64(a, b) ((a) == (b))
#define eq_f16(a, b) (__heq((a), (b)))
#define eq_bf16(a, b) (__heq((a), (b)))

#define DEFINE_REDUCE_KERNEL(rust_type, type, INIT_VAL)                                                                                                                                                                        \
    __device__ __forceinline__ void warpReduce_##rust_type(volatile type *sdata_##rust_type, volatile long long *sdata_##rust_type_idx, unsigned int tid)                                                                      \
    {                                                                                                                                                                                                                          \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 32]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 32]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 32]))      \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 32];                                                                                                                                                      \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 32];                                                                                                                                                              \
        }                                                                                                                                                                                                                      \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 16]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 16]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 16]))      \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 16];                                                                                                                                                      \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 16];                                                                                                                                                              \
        }                                                                                                                                                                                                                      \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 8]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 8]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 8]))         \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 8];                                                                                                                                                       \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 8];                                                                                                                                                               \
        }                                                                                                                                                                                                                      \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 4]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 4]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 4]))         \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 4];                                                                                                                                                       \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 4];                                                                                                                                                               \
        }                                                                                                                                                                                                                      \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 2]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 2]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 2]))         \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 2];                                                                                                                                                       \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 2];                                                                                                                                                               \
        }                                                                                                                                                                                                                      \
        if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 1]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 1]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + 1]))         \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 1];                                                                                                                                                       \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 1];                                                                                                                                                               \
        }                                                                                                                                                                                                                      \
    }                                                                                                                                                                                                                          \
    extern "C" __global__ void contiguous_reduce_##rust_type(type *out, long long *out_idx, type *in, size_t size)                                                                                                             \
    {                                                                                                                                                                                                                          \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                                                            \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                                                        \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                        \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                            \
        sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                                     \
        if (i + blockDim.x < size)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                      \
            if (gt_##rust_type(in[i], in[i + blockDim.x]))                                                                                                                                                                     \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[i + blockDim.x];                                                                                                                                                                   \
                sdata_##rust_type_idx[tid] = (long long)(i + blockDim.x);                                                                                                                                                      \
            }                                                                                                                                                                                                                  \
            else                                                                                                                                                                                                               \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[i];                                                                                                                                                                                \
                sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                   \
            }                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                      \
        else if (i < size)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type[tid] = in[i];                                                                                                                                                                                    \
            sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                       \
        }                                                                                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                                                       \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                               \
        {                                                                                                                                                                                                                      \
            if (tid < s)                                                                                                                                                                                                       \
            {                                                                                                                                                                                                                  \
                if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + s])) \
                {                                                                                                                                                                                                              \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                                                       \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                                               \
                }                                                                                                                                                                                                              \
            }                                                                                                                                                                                                                  \
            __syncthreads();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                      \
        if (tid < WRAP)                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                      \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                                             \
        }                                                                                                                                                                                                                      \
        if (tid == 0)                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                      \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                                                            \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                                                    \
        }                                                                                                                                                                                                                      \
    }                                                                                                                                                                                                                          \
    extern "C" __global__ void uncontiguous_reduce_##rust_type(type *out, long long *out_idx, type *in, long long *shape, long long *strides, size_t ndim, size_t size)                                                        \
    {                                                                                                                                                                                                                          \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                                                            \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                                                        \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                        \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                            \
        sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                                     \
        if (i + blockDim.x < size)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                      \
            long long a_offset = 0;                                                                                                                                                                                            \
            long long a_amount = i;                                                                                                                                                                                            \
            long long b_offset = 0;                                                                                                                                                                                            \
            long long b_amount = i + blockDim.x;                                                                                                                                                                               \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                                \
            {                                                                                                                                                                                                                  \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                                \
                a_amount /= shape[j];                                                                                                                                                                                          \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                                                \
                b_amount /= shape[j];                                                                                                                                                                                          \
            }                                                                                                                                                                                                                  \
            if (gt_##rust_type(in[a_offset], in[b_offset]))                                                                                                                                                                    \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                                         \
                sdata_##rust_type_idx[tid] = (long long)(i + blockDim.x);                                                                                                                                                      \
            }                                                                                                                                                                                                                  \
            else                                                                                                                                                                                                               \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                                         \
                sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                   \
            }                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                      \
        else if (i < size)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                      \
            long long a_offset = 0;                                                                                                                                                                                            \
            long long a_amount = i;                                                                                                                                                                                            \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                                \
            {                                                                                                                                                                                                                  \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                                \
                a_amount /= shape[j];                                                                                                                                                                                          \
            }                                                                                                                                                                                                                  \
            sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                                             \
            sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                       \
        }                                                                                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                                                       \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                               \
        {                                                                                                                                                                                                                      \
            if (tid < s)                                                                                                                                                                                                       \
            {                                                                                                                                                                                                                  \
                if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + s])) \
                {                                                                                                                                                                                                              \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                                                       \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                                               \
                }                                                                                                                                                                                                              \
            }                                                                                                                                                                                                                  \
            __syncthreads();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                      \
        if (tid < WRAP)                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                      \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                                             \
        }                                                                                                                                                                                                                      \
        if (tid == 0)                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                      \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                                                            \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                                                    \
        }                                                                                                                                                                                                                      \
    }                                                                                                                                                                                                                          \
    extern "C" __global__ void contiguous_reduce2_##rust_type(type *out, long long *out_idx, type *in, size_t size)                                                                                                            \
    {                                                                                                                                                                                                                          \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                                                            \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                                                        \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                        \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                            \
        sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                                     \
        if (i + blockDim.x < size)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                      \
            if (gt_##rust_type(in[i], in[i + blockDim.x]) || (eq_##rust_type(in[i], in[i + blockDim.x]) && out_idx[i] > out_idx[i + blockDim.x]))                                                                              \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[i + blockDim.x];                                                                                                                                                                   \
                sdata_##rust_type_idx[tid] = out_idx[i + blockDim.x];                                                                                                                                                          \
            }                                                                                                                                                                                                                  \
            else                                                                                                                                                                                                               \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[i];                                                                                                                                                                                \
                sdata_##rust_type_idx[tid] = out_idx[i];                                                                                                                                                                       \
            }                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                      \
        else if (i < size)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type[tid] = in[i];                                                                                                                                                                                    \
            sdata_##rust_type_idx[tid] = out_idx[i];                                                                                                                                                                           \
        }                                                                                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                                                       \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                               \
        {                                                                                                                                                                                                                      \
            if (tid < s)                                                                                                                                                                                                       \
            {                                                                                                                                                                                                                  \
                if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + s])) \
                {                                                                                                                                                                                                              \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                                                       \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                                               \
                }                                                                                                                                                                                                              \
            }                                                                                                                                                                                                                  \
            __syncthreads();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                      \
        if (tid < WRAP)                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                      \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                                             \
        }                                                                                                                                                                                                                      \
        if (tid == 0)                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                      \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                                                            \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                                                    \
        }                                                                                                                                                                                                                      \
    }                                                                                                                                                                                                                          \
    extern "C" __global__ void nkd_##rust_type(type *out, type *in, long long *out_idx, long long *shape, long long *strides, size_t ndim, size_t start_row_idx, size_t cols, size_t rows, size_t num_blocks_per_row)          \
    {                                                                                                                                                                                                                          \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                                                            \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                                                        \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                        \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                            \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                                     \
        sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                           \
        if (blockIdx.y + start_row_idx >= rows)                                                                                                                                                                                \
        {                                                                                                                                                                                                                      \
            return;                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                      \
        if (i + blockDim.x < cols)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                      \
            long long a_offset = 0;                                                                                                                                                                                            \
            long long a_amount = i + (blockIdx.y + start_row_idx) * cols;                                                                                                                                                      \
            long long b_offset = 0;                                                                                                                                                                                            \
            long long b_amount = i + blockDim.x + (blockIdx.y + start_row_idx) * cols;                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                                \
            {                                                                                                                                                                                                                  \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                                \
                a_amount /= shape[j];                                                                                                                                                                                          \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                                                \
                b_amount /= shape[j];                                                                                                                                                                                          \
            }                                                                                                                                                                                                                  \
            if (gt_##rust_type(in[a_offset], in[b_offset]))                                                                                                                                                                    \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[b_offset];                                                                                                                                                                         \
                sdata_##rust_type_idx[tid] = (long long)(i + blockDim.x);                                                                                                                                                      \
            }                                                                                                                                                                                                                  \
            else                                                                                                                                                                                                               \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                                         \
                sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                   \
            }                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                      \
        else if (i < cols)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                      \
            long long a_amount = i + (blockIdx.y + start_row_idx) * cols;                                                                                                                                                      \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                                \
            {                                                                                                                                                                                                                  \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                                                                      \
                a_amount /= shape[j];                                                                                                                                                                                          \
            }                                                                                                                                                                                                                  \
            sdata_##rust_type[tid] = *in;                                                                                                                                                                                      \
            sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                       \
        }                                                                                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                                                       \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                               \
        {                                                                                                                                                                                                                      \
            if (tid < s)                                                                                                                                                                                                       \
            {                                                                                                                                                                                                                  \
                if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + s])) \
                {                                                                                                                                                                                                              \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                                                       \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                                               \
                }                                                                                                                                                                                                              \
            }                                                                                                                                                                                                                  \
            __syncthreads();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                      \
        if (tid < WRAP)                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                      \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                                             \
        }                                                                                                                                                                                                                      \
        if (tid == 0)                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                      \
            out[blockIdx.x + (blockIdx.y + start_row_idx) * num_blocks_per_row] = sdata_##rust_type[0];                                                                                                                        \
            out_idx[blockIdx.x + (blockIdx.y + start_row_idx) * num_blocks_per_row] = sdata_##rust_type_idx[0];                                                                                                                \
        }                                                                                                                                                                                                                      \
    }                                                                                                                                                                                                                          \
    extern "C" __global__ void nkd2_##rust_type(type *out, type *in, long long *out_idx, size_t start_row_idx, size_t cols, size_t rows, size_t num_blocks_per_row, size_t original_cols)                                      \
    {                                                                                                                                                                                                                          \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                                                            \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                                                        \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                        \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                            \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                                     \
        sdata_##rust_type_idx[tid] = (long long)(i);                                                                                                                                                                           \
        if (blockIdx.y + start_row_idx >= rows)                                                                                                                                                                                \
        {                                                                                                                                                                                                                      \
            return;                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                      \
        unsigned int a_idx = i + (blockIdx.y + start_row_idx) * original_cols;                                                                                                                                                 \
        unsigned int b_idx = i + blockDim.x + (blockIdx.y + start_row_idx) * original_cols;                                                                                                                                    \
        if (i + blockDim.x < cols)                                                                                                                                                                                             \
        {                                                                                                                                                                                                                      \
            if (gt_##rust_type(in[a_idx], in[b_idx]) || (eq_##rust_type(in[a_idx], in[b_idx]) && out_idx[a_idx] > out_idx[b_idx]))                                                                                             \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[b_idx];                                                                                                                                                                            \
                sdata_##rust_type_idx[tid] = out_idx[b_idx];                                                                                                                                                                   \
            }                                                                                                                                                                                                                  \
            else                                                                                                                                                                                                               \
            {                                                                                                                                                                                                                  \
                sdata_##rust_type[tid] = in[a_idx];                                                                                                                                                                            \
                sdata_##rust_type_idx[tid] = out_idx[a_idx];                                                                                                                                                                   \
            }                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                      \
        else if (i < cols)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                      \
            sdata_##rust_type[tid] = in[a_idx];                                                                                                                                                                                \
            sdata_##rust_type_idx[tid] = out_idx[a_idx];                                                                                                                                                                       \
        }                                                                                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                                                       \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                               \
        {                                                                                                                                                                                                                      \
            if (tid < s)                                                                                                                                                                                                       \
            {                                                                                                                                                                                                                  \
                if (gt_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) || (eq_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]) && sdata_##rust_type_idx[tid] > sdata_##rust_type_idx[tid + s])) \
                {                                                                                                                                                                                                              \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                                                       \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                                               \
                }                                                                                                                                                                                                              \
            }                                                                                                                                                                                                                  \
            __syncthreads();                                                                                                                                                                                                   \
        }                                                                                                                                                                                                                      \
        if (tid < WRAP)                                                                                                                                                                                                        \
        {                                                                                                                                                                                                                      \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                                             \
        }                                                                                                                                                                                                                      \
        if (tid == 0)                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                      \
            out[blockIdx.x + (blockIdx.y + start_row_idx) * original_cols] = sdata_##rust_type[0];                                                                                                                             \
            out_idx[blockIdx.x + (blockIdx.y + start_row_idx) * original_cols] = sdata_##rust_type_idx[0];                                                                                                                     \
        }                                                                                                                                                                                                                      \
    }

DEFINE_REDUCE_KERNEL(bool, bool, BOOL_MAX)
DEFINE_REDUCE_KERNEL(i8, char, I8_MAX)
DEFINE_REDUCE_KERNEL(i16, short, I16_MAX)
DEFINE_REDUCE_KERNEL(i32, int, I32_MAX)
DEFINE_REDUCE_KERNEL(i64, long long, I64_MAX)
DEFINE_REDUCE_KERNEL(u8, unsigned char, U8_MAX)
DEFINE_REDUCE_KERNEL(u16, unsigned short, U16_MAX)
DEFINE_REDUCE_KERNEL(u32, unsigned int, U32_MAX)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, U64_MAX)
DEFINE_REDUCE_KERNEL(f32, float, F32_MAX)
DEFINE_REDUCE_KERNEL(f64, double, F64_MAX)
DEFINE_REDUCE_KERNEL(f16, __half, F16_MAX)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, BF16_MAX)
