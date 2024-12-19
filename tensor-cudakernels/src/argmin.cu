#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <limits.h>

#define WRAP 32

#define BOOL_MAX 1
#define I8_MAX SCHAR_MAX
#define I16_MAX SHRT_MAX
#define I32_MAX INT_MAX
#define I64_MAX LLONG_MAX
#define F32_MAX 1.0f / 0.0f
#define F64_MAX 1.0 / 0.0

#define U8_MAX UCHAR_MAX
#define U16_MAX USHRT_MAX
#define U32_MAX UINT_MAX
#define U64_MAX ULONG_MAX
#define F16_MAX __half((unsigned short)31744)
#define BF16_MAX __nv_bfloat16((unsigned short)0x7F80)

#define min_bool(a, b) ((bool)min(((unsigned char)a), ((unsigned char)b)))
#define min_i8(a, b) min((a), (b))
#define min_i16(a, b) min((a), (b))
#define min_i32(a, b) min((a), (b))
#define min_i64(a, b) min((a), (b))
#define min_f32(a, b) min((a), (b))
#define min_f64(a, b) min((a), (b))

#define min_u8(a, b) min((a), (b))
#define min_u16(a, b) min((a), (b))
#define min_u32(a, b) min((a), (b))
#define min_u64(a, b) min((a), (b))
#define min_f16(a, b) __hmin((a), (b))
#define min_bf16(a, b) __hmin((a), (b))

#define eq_bool(a, b) ((a) == (b))
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

#define DEFINE_REDUCE_KERNEL(rust_type, type, INIT_VAL)                                                                                                                                           \
    __device__ __forceinline__ void warpReduce_##rust_type(volatile type *sdata_##rust_type, volatile long long *sdata_##rust_type_idx, unsigned int tid)                                         \
    {                                                                                                                                                                                             \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 32]), sdata_##rust_type[tid + 32]))                                                                    \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 32];                                                                                                                         \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 32];                                                                                                                                 \
        }                                                                                                                                                                                         \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 16]), sdata_##rust_type[tid + 16]))                                                                    \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 16];                                                                                                                         \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 16];                                                                                                                                 \
        }                                                                                                                                                                                         \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 8]), sdata_##rust_type[tid + 8]))                                                                      \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 8];                                                                                                                          \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 8];                                                                                                                                  \
        }                                                                                                                                                                                         \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 4]), sdata_##rust_type[tid + 4]))                                                                      \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 4];                                                                                                                          \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 4];                                                                                                                                  \
        }                                                                                                                                                                                         \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 2]), sdata_##rust_type[tid + 2]))                                                                      \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 2];                                                                                                                          \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 2];                                                                                                                                  \
        }                                                                                                                                                                                         \
        if (eq_##rust_type(min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 1]), sdata_##rust_type[tid + 1]))                                                                      \
        {                                                                                                                                                                                         \
            sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + 1];                                                                                                                          \
            sdata_##rust_type[tid] = sdata_##rust_type[tid + 1];                                                                                                                                  \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce_##rust_type(type *out, long long *out_idx, type *in, size_t size)                                                                                \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                                           \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                               \
        sdata_##rust_type_idx[tid] = i;                                                                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        if (i + blockDim.x < size)                                                                                                                                                                \
        {                                                                                                                                                                                         \
            type min_val = min_##rust_type(in[i], in[i + blockDim.x]);                                                                                                                            \
            if (min_val == in[i + blockDim.x])                                                                                                                                                    \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i + blockDim.x];                                                                                                                                      \
                sdata_##rust_type_idx[tid] = i + blockDim.x;                                                                                                                                      \
            }                                                                                                                                                                                     \
            else                                                                                                                                                                                  \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i];                                                                                                                                                   \
            }                                                                                                                                                                                     \
        }                                                                                                                                                                                         \
        else if (i < size)                                                                                                                                                                        \
        {                                                                                                                                                                                         \
            sdata_##rust_type[tid] = in[i];                                                                                                                                                       \
        }                                                                                                                                                                                         \
        __syncthreads();                                                                                                                                                                          \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                  \
        {                                                                                                                                                                                         \
            if (tid < s)                                                                                                                                                                          \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                                               \
                if (min_val == sdata_##rust_type[tid + s])                                                                                                                                        \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                          \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                  \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            __syncthreads();                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        if (tid < WRAP)                                                                                                                                                                           \
        {                                                                                                                                                                                         \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                \
        }                                                                                                                                                                                         \
        if (tid == 0)                                                                                                                                                                             \
        {                                                                                                                                                                                         \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                               \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                       \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce2_##rust_type(type *out, long long *out_idx, type *in, long long *inp_idx, size_t size)                                                           \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                                           \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                               \
        sdata_##rust_type_idx[tid] = I64_MAX;                                                                                                                                                     \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        if (i + blockDim.x < size)                                                                                                                                                                \
        {                                                                                                                                                                                         \
            type min_val = min_##rust_type(in[i], in[i + blockDim.x]);                                                                                                                            \
            if (min_val == in[i + blockDim.x])                                                                                                                                                    \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i + blockDim.x];                                                                                                                                      \
                sdata_##rust_type_idx[tid] = inp_idx[i + blockDim.x];                                                                                                                             \
            }                                                                                                                                                                                     \
            else                                                                                                                                                                                  \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i];                                                                                                                                                   \
                sdata_##rust_type_idx[tid] = inp_idx[i];                                                                                                                                          \
            }                                                                                                                                                                                     \
        }                                                                                                                                                                                         \
        else if (i < size)                                                                                                                                                                        \
        {                                                                                                                                                                                         \
            sdata_##rust_type[tid] = in[i];                                                                                                                                                       \
            sdata_##rust_type_idx[tid] = inp_idx[i];                                                                                                                                              \
        }                                                                                                                                                                                         \
        __syncthreads();                                                                                                                                                                          \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                  \
        {                                                                                                                                                                                         \
            if (tid < s)                                                                                                                                                                          \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                                               \
                if (min_val == sdata_##rust_type[tid + s])                                                                                                                                        \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                          \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                  \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            __syncthreads();                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        if (tid < WRAP)                                                                                                                                                                           \
        {                                                                                                                                                                                         \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                \
        }                                                                                                                                                                                         \
        if (tid == 0)                                                                                                                                                                             \
        {                                                                                                                                                                                         \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                               \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                       \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void uncontiguous_reduce_##rust_type(type *out, long long *out_idx, type *in, long long *shape, long long *strides, size_t ndim, size_t size)                           \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                                           \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                               \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        sdata_##rust_type_idx[tid] = i;                                                                                                                                                           \
        if (i + blockDim.x < size)                                                                                                                                                                \
        {                                                                                                                                                                                         \
            long long a_amount = i;                                                                                                                                                               \
            long long b_amount = i + blockDim.x;                                                                                                                                                  \
            long long a_offset = 0;                                                                                                                                                               \
            long long b_offset = 0;                                                                                                                                                               \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                   \
            {                                                                                                                                                                                     \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                   \
                a_amount /= shape[j];                                                                                                                                                             \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                   \
                b_amount /= shape[j];                                                                                                                                                             \
            }                                                                                                                                                                                     \
            type min_val = min_##rust_type(in[a_offset], in[b_offset]);                                                                                                                           \
            if (min_val == in[b_offset])                                                                                                                                                          \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[b_offset];                                                                                                                                            \
                sdata_##rust_type_idx[tid] = i + blockDim.x;                                                                                                                                      \
            }                                                                                                                                                                                     \
            else                                                                                                                                                                                  \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[a_offset];                                                                                                                                            \
            }                                                                                                                                                                                     \
        }                                                                                                                                                                                         \
        else if (i < size)                                                                                                                                                                        \
        {                                                                                                                                                                                         \
            long long a_amount = i;                                                                                                                                                               \
            long long a_offset = 0;                                                                                                                                                               \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                   \
            {                                                                                                                                                                                     \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                   \
                a_amount /= shape[j];                                                                                                                                                             \
            }                                                                                                                                                                                     \
            sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                \
        }                                                                                                                                                                                         \
        __syncthreads();                                                                                                                                                                          \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                  \
        {                                                                                                                                                                                         \
            type min_val = min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                                                   \
            if (min_val == sdata_##rust_type[tid + s])                                                                                                                                            \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                              \
                sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                      \
            }                                                                                                                                                                                     \
            __syncthreads();                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        if (tid < WRAP)                                                                                                                                                                           \
        {                                                                                                                                                                                         \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                \
        }                                                                                                                                                                                         \
        if (tid == 0)                                                                                                                                                                             \
        {                                                                                                                                                                                         \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                                               \
            out_idx[blockIdx.x] = sdata_##rust_type_idx[0];                                                                                                                                       \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce3_##rust_type(type *out, long long *out_idx, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t num_blocks_per_row) \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                                           \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                               \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        sdata_##rust_type_idx[tid] = i;                                                                                                                                                           \
        if (i + blockDim.x < cols)                                                                                                                                                                \
        {                                                                                                                                                                                         \
            long long a_offset = 0;                                                                                                                                                               \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                                           \
            long long b_offset = 0;                                                                                                                                                               \
            long long b_amount = i + blockDim.x + blockIdx.y * cols;                                                                                                                              \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                   \
            {                                                                                                                                                                                     \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                   \
                a_amount /= shape[j];                                                                                                                                                             \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                   \
                b_amount /= shape[j];                                                                                                                                                             \
            }                                                                                                                                                                                     \
            type min_val = min_##rust_type(in[a_offset], in[b_offset]);                                                                                                                           \
            if (min_val == in[b_offset])                                                                                                                                                          \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[b_offset];                                                                                                                                            \
                sdata_##rust_type_idx[tid] = i + blockDim.x;                                                                                                                                      \
            }                                                                                                                                                                                     \
            else                                                                                                                                                                                  \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[a_offset];                                                                                                                                            \
            }                                                                                                                                                                                     \
        }                                                                                                                                                                                         \
        else if (i < cols)                                                                                                                                                                        \
        {                                                                                                                                                                                         \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                                           \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                   \
            {                                                                                                                                                                                     \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                                         \
                a_amount /= shape[j];                                                                                                                                                             \
            }                                                                                                                                                                                     \
            sdata_##rust_type[tid] = *in;                                                                                                                                                         \
        }                                                                                                                                                                                         \
        __syncthreads();                                                                                                                                                                          \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                  \
        {                                                                                                                                                                                         \
            if (tid < s)                                                                                                                                                                          \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                                               \
                if (min_val == sdata_##rust_type[tid + s])                                                                                                                                        \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                          \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                  \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            __syncthreads();                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        if (tid < WRAP)                                                                                                                                                                           \
        {                                                                                                                                                                                         \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                \
        }                                                                                                                                                                                         \
        if (tid == 0)                                                                                                                                                                             \
        {                                                                                                                                                                                         \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                                             \
            out_idx[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type_idx[0];                                                                                                     \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce33_##rust_type(type *out, long long *out_idx, type *in, long long *inp_idx, size_t cols, size_t num_blocks_per_row)                               \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x];                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                                           \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                               \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        sdata_##rust_type_idx[tid] = I64_MAX;                                                                                                                                                     \
        if (i + blockDim.x < cols)                                                                                                                                                                \
        {                                                                                                                                                                                         \
            type min_val = min_##rust_type(in[i + blockIdx.y * cols], in[i + blockDim.x + blockIdx.y * cols]);                                                                                    \
            if (min_val == in[i + blockDim.x + blockIdx.y * cols])                                                                                                                                \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i + blockIdx.y * cols];                                                                                                                               \
                sdata_##rust_type_idx[tid] = inp_idx[i + blockIdx.y * cols];                                                                                                                      \
            }                                                                                                                                                                                     \
            else                                                                                                                                                                                  \
            {                                                                                                                                                                                     \
                sdata_##rust_type[tid] = in[i + blockDim.x + blockIdx.y * cols];                                                                                                                  \
                sdata_##rust_type_idx[tid] = inp_idx[i + blockDim.x + blockIdx.y * cols];                                                                                                         \
            }                                                                                                                                                                                     \
        }                                                                                                                                                                                         \
        else if (i < cols)                                                                                                                                                                        \
        {                                                                                                                                                                                         \
            sdata_##rust_type[tid] = in[i + blockIdx.y * cols];                                                                                                                                   \
            sdata_##rust_type_idx[tid] = inp_idx[i + blockIdx.y * cols];                                                                                                                          \
        }                                                                                                                                                                                         \
        __syncthreads();                                                                                                                                                                          \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                  \
        {                                                                                                                                                                                         \
            if (tid < s)                                                                                                                                                                          \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                                               \
                if (min_val == sdata_##rust_type[tid + s])                                                                                                                                        \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[tid] = sdata_##rust_type[tid + s];                                                                                                                          \
                    sdata_##rust_type_idx[tid] = sdata_##rust_type_idx[tid + s];                                                                                                                  \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            __syncthreads();                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        if (tid < WRAP)                                                                                                                                                                           \
        {                                                                                                                                                                                         \
            warpReduce_##rust_type(sdata_##rust_type, sdata_##rust_type_idx, tid);                                                                                                                \
        }                                                                                                                                                                                         \
        if (tid == 0)                                                                                                                                                                             \
        {                                                                                                                                                                                         \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                                             \
            out_idx[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type_idx[0];                                                                                                     \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce4_##rust_type(type *out, long long *out_idx, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)               \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x * blockDim.y];                                                                                              \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                                \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                             \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                             \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        sdata_##rust_type_idx[tid] = row_idx * cols + col_idx;                                                                                                                                    \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                                   \
        {                                                                                                                                                                                         \
            return;                                                                                                                                                                               \
        }                                                                                                                                                                                         \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                              \
        long long offset = 0;                                                                                                                                                                     \
        for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                       \
        {                                                                                                                                                                                         \
            offset += (idx % shape[j]) * strides[j];                                                                                                                                              \
            idx /= shape[j];                                                                                                                                                                      \
        }                                                                                                                                                                                         \
        sdata_##rust_type[tid] = in[offset];                                                                                                                                                      \
        __syncthreads();                                                                                                                                                                          \
        if (threadIdx.y == 0)                                                                                                                                                                     \
        {                                                                                                                                                                                         \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                         \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[threadIdx.x], sdata_##rust_type[s * blockDim.x + threadIdx.x]);                                                                  \
                if (min_val == sdata_##rust_type[s * blockDim.x + threadIdx.x])                                                                                                                   \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[threadIdx.x] = sdata_##rust_type[s * blockDim.x + threadIdx.x];                                                                                             \
                    sdata_##rust_type_idx[threadIdx.x] = sdata_##rust_type_idx[s * blockDim.x + threadIdx.x];                                                                                     \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            out_idx[col_idx + blockIdx.y * cols] = sdata_##rust_type_idx[threadIdx.x];                                                                                                            \
            out[col_idx + blockIdx.y * cols] = sdata_##rust_type[threadIdx.x];                                                                                                                    \
        }                                                                                                                                                                                         \
    }                                                                                                                                                                                             \
    extern "C" __global__ void contiguous_reduce44_##rust_type(type *out, long long *out_idx, type *in, long long *inp_idx, size_t ndim, size_t cols, size_t rows)                                \
    {                                                                                                                                                                                             \
        extern __shared__ type sdata_##rust_type[];                                                                                                                                               \
        long long *sdata_##rust_type_idx = (long long *)&sdata_##rust_type[blockDim.x * blockDim.y];                                                                                              \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                                \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                             \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                             \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                        \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                                   \
        {                                                                                                                                                                                         \
            return;                                                                                                                                                                               \
        }                                                                                                                                                                                         \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                              \
        sdata_##rust_type[tid] = in[idx];                                                                                                                                                         \
        sdata_##rust_type_idx[tid] = inp_idx[idx];                                                                                                                                                \
        __syncthreads();                                                                                                                                                                          \
        if (threadIdx.y == 0)                                                                                                                                                                     \
        {                                                                                                                                                                                         \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                         \
            {                                                                                                                                                                                     \
                type min_val = min_##rust_type(sdata_##rust_type[threadIdx.x], sdata_##rust_type[s * blockDim.x + threadIdx.x]);                                                                  \
                if (min_val == sdata_##rust_type[s * blockDim.x + threadIdx.x])                                                                                                                   \
                {                                                                                                                                                                                 \
                    sdata_##rust_type[threadIdx.x] = sdata_##rust_type[s * blockDim.x + threadIdx.x];                                                                                             \
                    sdata_##rust_type_idx[threadIdx.x] = sdata_##rust_type_idx[s * blockDim.x + threadIdx.x];                                                                                     \
                }                                                                                                                                                                                 \
            }                                                                                                                                                                                     \
            out_idx[col_idx + blockIdx.y * cols] = sdata_##rust_type_idx[threadIdx.x];                                                                                                            \
            out[col_idx + blockIdx.y * cols] = sdata_##rust_type[threadIdx.x];                                                                                                                    \
        }                                                                                                                                                                                         \
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
