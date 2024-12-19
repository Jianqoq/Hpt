#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <limits.h>

#define WRAP 32

#define BOOL_MIN 0
#define I8_MIN SCHAR_MIN
#define I16_MIN SHRT_MIN
#define I32_MIN INT_MIN
#define I64_MIN LLONG_MIN
#define F32_MIN -1.0f / 0.0f
#define F64_MIN -1.0 / 0.0

#define U8_MIN 0
#define U16_MIN 0
#define U32_MIN 0
#define U64_MIN 0
#define F16_MIN __half((unsigned short)0xFC00U)
#define BF16_MIN __nv_bfloat16((unsigned short)0xFF80U)

#define max_bool(a, b) ((bool)max(((unsigned char)a), ((unsigned char)b)))
#define max_i8(a, b) max((a), (b))
#define max_i16(a, b) max((a), (b))
#define max_i32(a, b) max((a), (b))
#define max_i64(a, b) max((a), (b))
#define max_f32(a, b) max((a), (b))
#define max_f64(a, b) max((a), (b))

#define max_u8(a, b) max((a), (b))
#define max_u16(a, b) max((a), (b))
#define max_u32(a, b) max((a), (b))
#define max_u64(a, b) max((a), (b))

#define max_f16(a, b) __hmax((a), (b))
#define max_bf16(a, b) __hmax((a), (b))

#define DEFINE_REDUCE_KERNEL(rust_type, type, INIT_VAL)                                                                                                                       \
    __device__ __forceinline__ void warpReduce_##rust_type(volatile type *sdata_##rust_type, unsigned int tid)                                                                \
    {                                                                                                                                                                         \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 32]);                                                                        \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 16]);                                                                        \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 8]);                                                                         \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 4]);                                                                         \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 2]);                                                                         \
        sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 1]);                                                                         \
    }                                                                                                                                                                         \
    extern "C" __global__ void contiguous_reduce_##rust_type(type *out, type *in, size_t size)                                                                                \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                       \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (i + blockDim.x < size)                                                                                                                                            \
        {                                                                                                                                                                     \
            sdata_##rust_type[tid] = max_##rust_type(in[i], in[i + blockDim.x]);                                                                                              \
        }                                                                                                                                                                     \
        else if (i < size)                                                                                                                                                    \
        {                                                                                                                                                                     \
            sdata_##rust_type[tid] = in[i];                                                                                                                                   \
        }                                                                                                                                                                     \
        __syncthreads();                                                                                                                                                      \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                              \
        {                                                                                                                                                                     \
            if (tid < s)                                                                                                                                                      \
            {                                                                                                                                                                 \
                sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                 \
            }                                                                                                                                                                 \
            __syncthreads();                                                                                                                                                  \
        }                                                                                                                                                                     \
        if (tid < WRAP)                                                                                                                                                       \
        {                                                                                                                                                                     \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                                   \
        }                                                                                                                                                                     \
        if (tid == 0)                                                                                                                                                         \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                           \
    }                                                                                                                                                                         \
    extern "C" __global__ void uncontiguous_reduce_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t size)                           \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                       \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (i + blockDim.x < size)                                                                                                                                            \
        {                                                                                                                                                                     \
            long long a_amount = i;                                                                                                                                           \
            long long b_amount = i + blockDim.x;                                                                                                                              \
            long long a_offset = 0;                                                                                                                                           \
            long long b_offset = 0;                                                                                                                                           \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                               \
            {                                                                                                                                                                 \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                               \
                a_amount /= shape[j];                                                                                                                                         \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                               \
                b_amount /= shape[j];                                                                                                                                         \
            }                                                                                                                                                                 \
            sdata_##rust_type[tid] = max_##rust_type(in[a_offset], in[b_offset]);                                                                                             \
        }                                                                                                                                                                     \
        else if (i < size)                                                                                                                                                    \
        {                                                                                                                                                                     \
            long long a_amount = i;                                                                                                                                           \
            long long a_offset = 0;                                                                                                                                           \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                               \
            {                                                                                                                                                                 \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                               \
                a_amount /= shape[j];                                                                                                                                         \
            }                                                                                                                                                                 \
            sdata_##rust_type[tid] = in[a_offset];                                                                                                                            \
        }                                                                                                                                                                     \
        __syncthreads();                                                                                                                                                      \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                              \
        {                                                                                                                                                                     \
            if (tid < s)                                                                                                                                                      \
            {                                                                                                                                                                 \
                sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                 \
            }                                                                                                                                                                 \
            __syncthreads();                                                                                                                                                  \
        }                                                                                                                                                                     \
        if (tid < WRAP)                                                                                                                                                       \
        {                                                                                                                                                                     \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                                   \
        }                                                                                                                                                                     \
        if (tid == 0)                                                                                                                                                         \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                           \
    }                                                                                                                                                                         \
    extern "C" __global__ void contiguous_reduce2_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t num_blocks_per_row) \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                       \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (i + blockDim.x < cols)                                                                                                                                            \
        {                                                                                                                                                                     \
            long long a_offset = 0;                                                                                                                                           \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                       \
            long long b_offset = 0;                                                                                                                                           \
            long long b_amount = i + blockDim.x + blockIdx.y * cols;                                                                                                          \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                               \
            {                                                                                                                                                                 \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                               \
                a_amount /= shape[j];                                                                                                                                         \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                               \
                b_amount /= shape[j];                                                                                                                                         \
            }                                                                                                                                                                 \
            sdata_##rust_type[tid] = max_##rust_type(in[a_offset], in[b_offset]);                                                                                             \
        }                                                                                                                                                                     \
        else if (i < cols)                                                                                                                                                    \
        {                                                                                                                                                                     \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                       \
                                                                                                                                                                              \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                               \
            {                                                                                                                                                                 \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                     \
                a_amount /= shape[j];                                                                                                                                         \
            }                                                                                                                                                                 \
            sdata_##rust_type[tid] = *in;                                                                                                                                     \
        }                                                                                                                                                                     \
        __syncthreads();                                                                                                                                                      \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                              \
        {                                                                                                                                                                     \
            if (tid < s)                                                                                                                                                      \
            {                                                                                                                                                                 \
                sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                 \
            }                                                                                                                                                                 \
            __syncthreads();                                                                                                                                                  \
        }                                                                                                                                                                     \
        if (tid < WRAP)                                                                                                                                                       \
        {                                                                                                                                                                     \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                                   \
        }                                                                                                                                                                     \
        if (tid == 0)                                                                                                                                                         \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                         \
    }                                                                                                                                                                         \
                                                                                                                                                                              \
    extern "C" __global__ void contiguous_reduce22_##rust_type(type *out, type *in, size_t cols, size_t num_blocks_per_row)                                                   \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.x;                                                                                                                                       \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                           \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (i + blockDim.x < cols)                                                                                                                                            \
        {                                                                                                                                                                     \
            sdata_##rust_type[tid] = max_##rust_type(in[i + blockIdx.y * cols], in[i + blockDim.x + blockIdx.y * cols]);                                                      \
        }                                                                                                                                                                     \
        else if (i < cols)                                                                                                                                                    \
        {                                                                                                                                                                     \
            sdata_##rust_type[tid] = in[i + blockIdx.y * cols];                                                                                                               \
        }                                                                                                                                                                     \
        __syncthreads();                                                                                                                                                      \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                              \
        {                                                                                                                                                                     \
            if (tid < s)                                                                                                                                                      \
            {                                                                                                                                                                 \
                sdata_##rust_type[tid] = max_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]);                                                                 \
            }                                                                                                                                                                 \
            __syncthreads();                                                                                                                                                  \
        }                                                                                                                                                                     \
        if (tid < WRAP)                                                                                                                                                       \
        {                                                                                                                                                                     \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                                   \
        }                                                                                                                                                                     \
        if (tid == 0)                                                                                                                                                         \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                         \
    }                                                                                                                                                                         \
    extern "C" __global__ void contiguous_reduce3_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)               \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                            \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                         \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                         \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                               \
        {                                                                                                                                                                     \
            return;                                                                                                                                                           \
        }                                                                                                                                                                     \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                          \
        long long offset = 0;                                                                                                                                                 \
        for (int j = ndim - 1; j >= 0; j--)                                                                                                                                   \
        {                                                                                                                                                                     \
            offset += (idx % shape[j]) * strides[j];                                                                                                                          \
            idx /= shape[j];                                                                                                                                                  \
        }                                                                                                                                                                     \
        sdata_##rust_type[tid] = in[offset];                                                                                                                                  \
        __syncthreads();                                                                                                                                                      \
        if (threadIdx.y == 0)                                                                                                                                                 \
        {                                                                                                                                                                     \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                     \
            {                                                                                                                                                                 \
                sdata_##rust_type[threadIdx.x] = max_##rust_type(sdata_##rust_type[threadIdx.x], sdata_##rust_type[s * blockDim.x + threadIdx.x]);                            \
            }                                                                                                                                                                 \
            out[col_idx + blockIdx.y * cols] = sdata_##rust_type[threadIdx.x];                                                                                                \
        }                                                                                                                                                                     \
    }                                                                                                                                                                         \
    extern "C" __global__ void contiguous_reduce33_##rust_type(type *out, type *in, size_t ndim, size_t cols, size_t rows)                                                    \
    {                                                                                                                                                                         \
        extern __shared__ type sdata_##rust_type[];                                                                                                                           \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                            \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                         \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                         \
        sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                    \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                               \
        {                                                                                                                                                                     \
            return;                                                                                                                                                           \
        }                                                                                                                                                                     \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                          \
        sdata_##rust_type[tid] = in[idx];                                                                                                                                     \
        __syncthreads();                                                                                                                                                      \
        if (threadIdx.y == 0)                                                                                                                                                 \
        {                                                                                                                                                                     \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                     \
            {                                                                                                                                                                 \
                sdata_##rust_type[threadIdx.x] = max_##rust_type(sdata_##rust_type[threadIdx.x], sdata_##rust_type[s * blockDim.x + threadIdx.x]);                            \
            }                                                                                                                                                                 \
            out[col_idx + blockIdx.y * cols] = sdata_##rust_type[threadIdx.x];                                                                                                \
        }                                                                                                                                                                     \
    }

DEFINE_REDUCE_KERNEL(bool, bool, BOOL_MIN)
DEFINE_REDUCE_KERNEL(i8, char, I8_MIN)
DEFINE_REDUCE_KERNEL(i16, short, I16_MIN)
DEFINE_REDUCE_KERNEL(i32, int, I32_MIN)
DEFINE_REDUCE_KERNEL(i64, long long, I64_MIN)
DEFINE_REDUCE_KERNEL(u8, unsigned char, U8_MIN)
DEFINE_REDUCE_KERNEL(u16, unsigned short, U16_MIN)
DEFINE_REDUCE_KERNEL(u32, unsigned int, U32_MIN)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, U64_MIN)
DEFINE_REDUCE_KERNEL(f32, float, F32_MIN)
DEFINE_REDUCE_KERNEL(f64, double, F64_MIN)
DEFINE_REDUCE_KERNEL(f16, __half, F16_MIN)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, BF16_MIN)
