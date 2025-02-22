/*
reduce2 is similar to reduce.cu, reduce2 uses 2 level op
*/

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define WRAP 32

#define DEFINE_REDUCE_KERNEL(rust_type, in_type, out_type, INIT_VAL, METHOD)                                                                                                                                                \
    __device__ __forceinline__ void warpReduce_##METHOD##_##rust_type(volatile out_type *METHOD##sdata_##rust_type, unsigned int tid)                                                                                       \
    {                                                                                                                                                                                                                       \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP]);                                                                                      \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 2]);                                                                                  \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 4]);                                                                                  \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 8]);                                                                                  \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 16]);                                                                                 \
        METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 32]);                                                                                 \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void contiguous_##METHOD##_##rust_type(out_type *out, in_type *in, size_t size)                                                                                                                   \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (i + blockDim.x < size)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = METHOD##1_##rust_type(in[i], in[i + blockDim.x]);                                                                                                                              \
        }                                                                                                                                                                                                                   \
        else if (i < size)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = METHOD##1_unary_##rust_type(in[i]);                                                                                                                                            \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                                                                 \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void contiguous_cumulate_##METHOD##_##rust_type(out_type *out, out_type *in, size_t size)                                                                                                         \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (i + blockDim.x < size)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(in[i], in[i + blockDim.x]);                                                                                                                              \
        }                                                                                                                                                                                                                   \
        else if (i < size)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = in[i];                                                                                                                                                                         \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                                                                 \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void uncontiguous_##METHOD##_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t size)                                                              \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (i + blockDim.x < size)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            long long a_amount = i;                                                                                                                                                                                         \
            long long b_amount = i + blockDim.x;                                                                                                                                                                            \
            long long a_offset = 0;                                                                                                                                                                                         \
            long long b_offset = 0;                                                                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                             \
                a_amount /= shape[j];                                                                                                                                                                                       \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                                             \
                b_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = METHOD##1_##rust_type(in[a_offset], in[b_offset]);                                                                                                                             \
        }                                                                                                                                                                                                                   \
        else if (i < size)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            long long a_amount = i;                                                                                                                                                                                         \
            long long a_offset = 0;                                                                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                             \
                a_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = METHOD##1_unary_##rust_type(in[a_offset]);                                                                                                                                     \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                                                                 \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void uncontiguous_cumulate_##METHOD##_##rust_type(out_type *out, out_type *in, long long *shape, long long *strides, size_t ndim, size_t size)                                                    \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (i + blockDim.x < size)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            long long a_amount = i;                                                                                                                                                                                         \
            long long b_amount = i + blockDim.x;                                                                                                                                                                            \
            long long a_offset = 0;                                                                                                                                                                                         \
            long long b_offset = 0;                                                                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                             \
                a_amount /= shape[j];                                                                                                                                                                                       \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                                             \
                b_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(in[a_offset], in[b_offset]);                                                                                                                             \
        }                                                                                                                                                                                                                   \
        else if (i < size)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            long long a_amount = i;                                                                                                                                                                                         \
            long long a_offset = 0;                                                                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                             \
                a_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = in[a_offset];                                                                                                                                                                  \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                                                                 \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void contiguous_##METHOD##2_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t start_row_idx, size_t cols, size_t rows, size_t num_blocks_per_row) \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (blockIdx.y + start_row_idx >= rows)                                                                                                                                                                             \
        {                                                                                                                                                                                                                   \
            return;                                                                                                                                                                                                         \
        }                                                                                                                                                                                                                   \
        if (i + blockDim.x < cols)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            long long a_offset = 0;                                                                                                                                                                                         \
            long long a_amount = i + (blockIdx.y + start_row_idx) * cols;                                                                                                                                                   \
            long long b_offset = 0;                                                                                                                                                                                         \
            long long b_amount = i + blockDim.x + (blockIdx.y + start_row_idx) * cols;                                                                                                                                      \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                                                             \
                a_amount /= shape[j];                                                                                                                                                                                       \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                                                             \
                b_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = METHOD##1_##rust_type(in[a_offset], in[b_offset]);                                                                                                                             \
        }                                                                                                                                                                                                                   \
        else if (i < cols)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            long long a_amount = i + (blockIdx.y + start_row_idx) * cols;                                                                                                                                                   \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                             \
            {                                                                                                                                                                                                               \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                                                                   \
                a_amount /= shape[j];                                                                                                                                                                                       \
            }                                                                                                                                                                                                               \
            METHOD##sdata_##rust_type[tid] = METHOD##1_unary_##rust_type(*in);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x + (blockIdx.y + start_row_idx) * num_blocks_per_row] = METHOD##sdata_##rust_type[0];                                                                                                             \
    }                                                                                                                                                                                                                       \
                                                                                                                                                                                                                            \
    extern "C" __global__ void contiguous_##METHOD##22_##rust_type(out_type *out, out_type *in, size_t start_row_idx, size_t cols, size_t rows, size_t num_blocks_per_row)                                                  \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.x;                                                                                                                                                                                     \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                         \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (blockIdx.y + start_row_idx >= rows)                                                                                                                                                                             \
        {                                                                                                                                                                                                                   \
            return;                                                                                                                                                                                                         \
        }                                                                                                                                                                                                                   \
        if (i + blockDim.x < cols)                                                                                                                                                                                          \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(in[i + (blockIdx.y + start_row_idx) * cols], in[i + blockDim.x + (blockIdx.y + start_row_idx) * cols]);                                                  \
        }                                                                                                                                                                                                                   \
        else if (i < cols)                                                                                                                                                                                                  \
        {                                                                                                                                                                                                                   \
            METHOD##sdata_##rust_type[tid] = in[i + (blockIdx.y + start_row_idx) * cols];                                                                                                                                   \
        }                                                                                                                                                                                                                   \
        __syncthreads();                                                                                                                                                                                                    \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                                                            \
        {                                                                                                                                                                                                                   \
            if (tid < s)                                                                                                                                                                                                    \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[tid] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                                                                 \
            }                                                                                                                                                                                                               \
            __syncthreads();                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        if (tid < WRAP)                                                                                                                                                                                                     \
        {                                                                                                                                                                                                                   \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                                                              \
        }                                                                                                                                                                                                                   \
        if (tid == 0)                                                                                                                                                                                                       \
            out[blockIdx.x + (blockIdx.y + start_row_idx) * num_blocks_per_row] = METHOD##sdata_##rust_type[0];                                                                                                             \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void contiguous_##METHOD##3_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)                                                  \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                                                          \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                       \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                                                       \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                                                             \
        {                                                                                                                                                                                                                   \
            return;                                                                                                                                                                                                         \
        }                                                                                                                                                                                                                   \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                                                        \
        long long offset = 0;                                                                                                                                                                                               \
        for (int j = ndim - 1; j >= 0; j--)                                                                                                                                                                                 \
        {                                                                                                                                                                                                                   \
            offset += (idx % shape[j]) * strides[j];                                                                                                                                                                        \
            idx /= shape[j];                                                                                                                                                                                                \
        }                                                                                                                                                                                                                   \
        METHOD##sdata_##rust_type[tid] = METHOD##1_unary_##rust_type(in[offset]);                                                                                                                                           \
        __syncthreads();                                                                                                                                                                                                    \
        if (threadIdx.y == 0)                                                                                                                                                                                               \
        {                                                                                                                                                                                                                   \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                                                   \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[threadIdx.x] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[threadIdx.x], METHOD##sdata_##rust_type[s * blockDim.x + threadIdx.x]);                                            \
            }                                                                                                                                                                                                               \
            out[col_idx + blockIdx.y * cols] = METHOD##sdata_##rust_type[threadIdx.x];                                                                                                                                      \
        }                                                                                                                                                                                                                   \
    }                                                                                                                                                                                                                       \
    extern "C" __global__ void contiguous_##METHOD##33_##rust_type(out_type *out, out_type *in, size_t ndim, size_t cols, size_t rows)                                                                                      \
    {                                                                                                                                                                                                                       \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                                                             \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                                                          \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                                                       \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                                                       \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                                                          \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                                                             \
        {                                                                                                                                                                                                                   \
            return;                                                                                                                                                                                                         \
        }                                                                                                                                                                                                                   \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                                                        \
        METHOD##sdata_##rust_type[tid] = in[idx];                                                                                                                                                                           \
        __syncthreads();                                                                                                                                                                                                    \
        if (threadIdx.y == 0)                                                                                                                                                                                               \
        {                                                                                                                                                                                                                   \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                                                   \
            {                                                                                                                                                                                                               \
                METHOD##sdata_##rust_type[threadIdx.x] = METHOD##2_##rust_type(METHOD##sdata_##rust_type[threadIdx.x], METHOD##sdata_##rust_type[s * blockDim.x + threadIdx.x]);                                            \
            }                                                                                                                                                                                                               \
            out[col_idx + blockIdx.y * cols] = METHOD##sdata_##rust_type[threadIdx.x];                                                                                                                                      \
        }                                                                                                                                                                                                                   \
    }

// sum_square
#define sum_square1_bool(a, b) ((bool)(((unsigned int)(a)) * ((unsigned int)(a)) + ((unsigned int)(b)) * ((unsigned int)(b))))
#define sum_square1_i8(a, b) (a) * (a) + (b) * (b)
#define sum_square1_i16(a, b) (a) * (a) + (b) * (b)
#define sum_square1_i32(a, b) (a) * (a) + (b) * (b)
#define sum_square1_i64(a, b) (a) * (a) + (b) * (b)
#define sum_square1_f32(a, b) (a) * (a) + (b) * (b)
#define sum_square1_f64(a, b) (a) * (a) + (b) * (b)

#define sum_square1_u8(a, b) (a) * (a) + (b) * (b)
#define sum_square1_u16(a, b) (a) * (a) + (b) * (b)
#define sum_square1_u32(a, b) (a) * (a) + (b) * (b)
#define sum_square1_u64(a, b) (a) * (a) + (b) * (b)

#define sum_square1_f16(a, b) (a) * (a) + (b) * (b)
#define sum_square1_bf16(a, b) (a) * (a) + (b) * (b)

#define sum_square1_unary_bool(a) ((bool)(((unsigned int)(a)) * ((unsigned int)(a))))
#define sum_square1_unary_i8(a) (a) * (a)
#define sum_square1_unary_i16(a) (a) * (a)
#define sum_square1_unary_i32(a) (a) * (a)
#define sum_square1_unary_i64(a) (a) * (a)
#define sum_square1_unary_f32(a) (a) * (a)
#define sum_square1_unary_f64(a) (a) * (a)

#define sum_square1_unary_u8(a) (a) * (a)
#define sum_square1_unary_u16(a) (a) * (a)
#define sum_square1_unary_u32(a) (a) * (a)
#define sum_square1_unary_u64(a) (a) * (a)

#define sum_square1_unary_f16(a) (a) * (a)
#define sum_square1_unary_bf16(a) (a) * (a)

#define sum_square2_bool(a, b) ((bool)(((unsigned int)(a)) + ((unsigned int)(b))))
#define sum_square2_i8(a, b) (a) + (b)
#define sum_square2_i16(a, b) (a) + (b)
#define sum_square2_i32(a, b) (a) + (b)
#define sum_square2_i64(a, b) (a) + (b)
#define sum_square2_f32(a, b) (a) + (b)
#define sum_square2_f64(a, b) (a) + (b)

#define sum_square2_u8(a, b) (a) + (b)
#define sum_square2_u16(a, b) (a) + (b)
#define sum_square2_u32(a, b) (a) + (b)
#define sum_square2_u64(a, b) (a) + (b)

#define sum_square2_f16(a, b) __hadd((a), (b))
#define sum_square2_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, bool, 0, sum_square)
DEFINE_REDUCE_KERNEL(i8, char, char, 0, sum_square)
DEFINE_REDUCE_KERNEL(i16, short, short, 0, sum_square)
DEFINE_REDUCE_KERNEL(i32, int, int, 0, sum_square)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 0, sum_square)

DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 0, sum_square)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 0, sum_square)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 0, sum_square)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 0, sum_square)

DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, sum_square)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, sum_square)

DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, sum_square)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, sum_square)

// reducel3
#define reducel31_bool(a, b) __float2half_rn(powf((float)(a), 3.0f) + powf((float)(b), 3.0f))
#define reducel31_i8(a, b) __float2half_rn(powf((float)abs(a), 3.0f) + powf((float)abs(b), 3.0f))
#define reducel31_i16(a, b) __float2half_rn(powf((float)abs(a), 3.0f) + powf((float)abs(b), 3.0f))
#define reducel31_i32(a, b) powf((float)abs(a), 3.0f) + powf((float)abs(b), 3.0f)
#define reducel31_i64(a, b) pow((double)abs(a), 3.0) + pow((double)abs(b), 3.0)
#define reducel31_f32(a, b) powf(abs(a), 3.0f) + powf(abs(b), 3.0f)
#define reducel31_f64(a, b) pow(abs(a), 3.0) + pow(abs(b), 3.0)

#define reducel31_u8(a, b) __float2half_rn(powf((float)(a), 3.0f) + powf((float)(b), 3.0f))
#define reducel31_u16(a, b) __float2half_rn(powf((float)(a), 3.0f) + powf((float)(b), 3.0f))
#define reducel31_u32(a, b) powf((float)(a), 3.0f) + powf((float)(b), 3.0f)
#define reducel31_u64(a, b) pow((double)(a), 3.0) + pow((double)(b), 3.0)

#define reducel31_f16(a, b) __float2half_rn(powf(abs(__half2float(a)), 3.0f) + powf(abs(__half2float(b)), 3.0f))
#define reducel31_bf16(a, b) __float2bfloat16_rn(powf(abs(__bfloat162float(a)), 3.0f) + powf(abs(__bfloat162float(b)), 3.0f))

#define reducel31_unary_bool(a) __float2half_rn(powf((float)(a), 3.0f))
#define reducel31_unary_i8(a) __float2half_rn(powf((float)abs(a), 3.0f))
#define reducel31_unary_i16(a) __float2half_rn(powf((float)abs(a), 3.0f))
#define reducel31_unary_i32(a) powf((float)abs(a), 3.0f)
#define reducel31_unary_i64(a) pow((double)abs(a), 3.0)
#define reducel31_unary_f32(a) powf(abs(a), 3.0f)
#define reducel31_unary_f64(a) pow(abs(a), 3.0)

#define reducel31_unary_u8(a) __float2half_rn(powf((float)(a), 3.0f))
#define reducel31_unary_u16(a) __float2half_rn(powf((float)(a), 3.0f))
#define reducel31_unary_u32(a) powf((float)(a), 3.0f)
#define reducel31_unary_u64(a) pow((double)(a), 3.0)

#define reducel31_unary_f16(a) __float2half_rn(powf(abs(__half2float(a)), 3.0f))
#define reducel31_unary_bf16(a) __float2bfloat16_rn(powf(abs(__bfloat162float(a)), 3.0f))

#define reducel32_bool(a, b) __hadd((a), (b))
#define reducel32_i8(a, b) __hadd((a), (b))
#define reducel32_i16(a, b) __hadd((a), (b))
#define reducel32_i32(a, b) (a) + (b)
#define reducel32_i64(a, b) (a) + (b)
#define reducel32_f32(a, b) (a) + (b)
#define reducel32_f64(a, b) (a) + (b)

#define reducel32_u8(a, b) __hadd((a), (b))
#define reducel32_u16(a, b) __hadd((a), (b))
#define reducel32_u32(a, b) (a) + (b)
#define reducel32_u64(a, b) (a) + (b)

#define reducel32_f16(a, b) __hadd((a), (b))
#define reducel32_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(i8, char, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(i16, short, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(i32, int, float, 0, reducel3)
DEFINE_REDUCE_KERNEL(i64, long long, double, 0, reducel3)

DEFINE_REDUCE_KERNEL(u8, unsigned char, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(u16, unsigned short, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(u32, unsigned int, float, 0, reducel3)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, double, 0, reducel3)

DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, reducel3)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, reducel3)

DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, reducel3)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, reducel3)

// mean
#define mean1_bool(a, b) __float2half_rn((float)(a) + (float)(b))
#define mean1_i8(a, b) __float2half_rn((float)(a + b))
#define mean1_i16(a, b) __float2half_rn((float)(a + b))
#define mean1_i32(a, b) (float)(a + b)
#define mean1_i64(a, b) (double)(a + b)
#define mean1_f32(a, b) (a) + (b)
#define mean1_f64(a, b) (a) + (b)

#define mean1_u8(a, b) __float2half_rn((float)(a + b))
#define mean1_u16(a, b) __float2half_rn((float)(a + b))
#define mean1_u32(a, b) (float)(a + b)
#define mean1_u64(a, b) (double)(a + b)

#define mean1_f16(a, b) __hadd((a), (b))
#define mean1_bf16(a, b) __hadd((a), (b))

#define mean1_unary_bool(a) __float2half_rn((float)(a))
#define mean1_unary_i8(a) __float2half_rn((float)(a))
#define mean1_unary_i16(a) __float2half_rn((float)(a))
#define mean1_unary_i32(a) (float)(a)
#define mean1_unary_i64(a) (double)(a)
#define mean1_unary_f32(a) (a)
#define mean1_unary_f64(a) (a)

#define mean1_unary_u8(a) __float2half_rn((float)(a))
#define mean1_unary_u16(a) __float2half_rn((float)(a))
#define mean1_unary_u32(a) (float)(a)
#define mean1_unary_u64(a) (double)(a)

#define mean1_unary_f16(a) (a)
#define mean1_unary_bf16(a) (a)

#define mean2_bool(a, b) __hadd((a), (b))
#define mean2_i8(a, b) __hadd((a), (b))
#define mean2_i16(a, b) __hadd((a), (b))
#define mean2_i32(a, b) (a) + (b)
#define mean2_i64(a, b) (a) + (b)
#define mean2_f32(a, b) (a) + (b)
#define mean2_f64(a, b) (a) + (b)

#define mean2_u8(a, b) __hadd((a), (b))
#define mean2_u16(a, b) __hadd((a), (b))
#define mean2_u32(a, b) (a) + (b)
#define mean2_u64(a, b) (a) + (b)

#define mean2_f16(a, b) __hadd((a), (b))
#define mean2_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, __half, 0, mean)
DEFINE_REDUCE_KERNEL(i8, char, __half, 0, mean)
DEFINE_REDUCE_KERNEL(i16, short, __half, 0, mean)
DEFINE_REDUCE_KERNEL(i32, int, float, 0, mean)
DEFINE_REDUCE_KERNEL(i64, long long, double, 0, mean)

DEFINE_REDUCE_KERNEL(u8, unsigned char, __half, 0, mean)
DEFINE_REDUCE_KERNEL(u16, unsigned short, __half, 0, mean)
DEFINE_REDUCE_KERNEL(u32, unsigned int, float, 0, mean)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, double, 0, mean)

DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, mean)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, mean)

DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, mean)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, mean)

// any
#define any1_bool(a, b) (a) || (b)
#define any1_i8(a, b) (a) || (b)
#define any1_i16(a, b) (a) || (b)
#define any1_i32(a, b) (a) || (b)
#define any1_i64(a, b) (a) || (b)
#define any1_f32(a, b) (a) || (b)
#define any1_f64(a, b) (a) || (b)

#define any1_u8(a, b) (a) || (b)
#define any1_u16(a, b) (a) || (b)
#define any1_u32(a, b) (a) || (b)
#define any1_u64(a, b) (a) || (b)

#define any1_f16(a, b) (a) || (b)
#define any1_bf16(a, b) (a) || (b)

#define any1_unary_bool(a) (a)
#define any1_unary_i8(a) ((bool)a)
#define any1_unary_i16(a) ((bool)a)
#define any1_unary_i32(a) ((bool)a)
#define any1_unary_i64(a) ((bool)a)
#define any1_unary_f32(a) ((bool)a)
#define any1_unary_f64(a) ((bool)a)

#define any1_unary_u8(a) ((bool)a)
#define any1_unary_u16(a) ((bool)a)
#define any1_unary_u32(a) ((bool)a)
#define any1_unary_u64(a) ((bool)a)

#define any1_unary_f16(a) ((bool)__half2float(a))
#define any1_unary_bf16(a) ((bool)__bfloat162float(a))

#define any2_bool(a, b) (a) || (b)
#define any2_i8(a, b) (a) || (b)
#define any2_i16(a, b) (a) || (b)
#define any2_i32(a, b) (a) || (b)
#define any2_i64(a, b) (a) || (b)
#define any2_f32(a, b) (a) || (b)
#define any2_f64(a, b) (a) || (b)

#define any2_u8(a, b) (a) || (b)
#define any2_u16(a, b) (a) || (b)
#define any2_u32(a, b) (a) || (b)
#define any2_u64(a, b) (a) || (b)

#define any2_f16(a, b) (a) || (b)
#define any2_bf16(a, b) (a) || (b)

DEFINE_REDUCE_KERNEL(bool, bool, bool, false, any)
DEFINE_REDUCE_KERNEL(i8, char, bool, false, any)
DEFINE_REDUCE_KERNEL(i16, short, bool, false, any)
DEFINE_REDUCE_KERNEL(i32, int, bool, false, any)
DEFINE_REDUCE_KERNEL(i64, long long, bool, false, any)

DEFINE_REDUCE_KERNEL(u8, unsigned char, bool, false, any)
DEFINE_REDUCE_KERNEL(u16, unsigned short, bool, false, any)
DEFINE_REDUCE_KERNEL(u32, unsigned int, bool, false, any)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, bool, false, any)

DEFINE_REDUCE_KERNEL(f32, float, bool, false, any)
DEFINE_REDUCE_KERNEL(f64, double, bool, false, any)

DEFINE_REDUCE_KERNEL(f16, __half, bool, false, any)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, bool, false, any)

// all
#define all1_bool(a, b) (a) && (b)
#define all1_i8(a, b) (a) && (b)
#define all1_i16(a, b) (a) && (b)
#define all1_i32(a, b) (a) && (b)
#define all1_i64(a, b) (a) && (b)
#define all1_f32(a, b) (a) && (b)
#define all1_f64(a, b) (a) && (b)

#define all1_u8(a, b) (a) && (b)
#define all1_u16(a, b) (a) && (b)
#define all1_u32(a, b) (a) && (b)
#define all1_u64(a, b) (a) && (b)

#define all1_f16(a, b) (a) && (b)
#define all1_bf16(a, b) (a) && (b)

#define all1_unary_bool(a) (a)
#define all1_unary_i8(a) ((bool)a)
#define all1_unary_i16(a) ((bool)a)
#define all1_unary_i32(a) ((bool)a)
#define all1_unary_i64(a) ((bool)a)
#define all1_unary_f32(a) ((bool)a)
#define all1_unary_f64(a) ((bool)a)

#define all1_unary_u8(a) ((bool)a)
#define all1_unary_u16(a) ((bool)a)
#define all1_unary_u32(a) ((bool)a)
#define all1_unary_u64(a) ((bool)a)

#define all1_unary_f16(a) ((bool)__half2float(a))
#define all1_unary_bf16(a) ((bool)__bfloat162float(a))

#define all2_bool(a, b) (a) && (b)
#define all2_i8(a, b) (a) && (b)
#define all2_i16(a, b) (a) && (b)
#define all2_i32(a, b) (a) && (b)
#define all2_i64(a, b) (a) && (b)
#define all2_f32(a, b) (a) && (b)
#define all2_f64(a, b) (a) && (b)

#define all2_u8(a, b) (a) && (b)
#define all2_u16(a, b) (a) && (b)
#define all2_u32(a, b) (a) && (b)
#define all2_u64(a, b) (a) && (b)

#define all2_f16(a, b) (a) && (b)
#define all2_bf16(a, b) (a) && (b)

DEFINE_REDUCE_KERNEL(bool, bool, bool, true, all)
DEFINE_REDUCE_KERNEL(i8, char, bool, true, all)
DEFINE_REDUCE_KERNEL(i16, short, bool, true, all)
DEFINE_REDUCE_KERNEL(i32, int, bool, true, all)
DEFINE_REDUCE_KERNEL(i64, long long, bool, true, all)

DEFINE_REDUCE_KERNEL(u8, unsigned char, bool, true, all)
DEFINE_REDUCE_KERNEL(u16, unsigned short, bool, true, all)
DEFINE_REDUCE_KERNEL(u32, unsigned int, bool, true, all)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, bool, true, all)

DEFINE_REDUCE_KERNEL(f32, float, bool, true, all)
DEFINE_REDUCE_KERNEL(f64, double, bool, true, all)

DEFINE_REDUCE_KERNEL(f16, __half, bool, true, all)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, bool, true, all)

// logsumexp
#define logsumexp1_bool(a, b) __float2half_rn(expf((float)(a)) + expf((float)(b)))
#define logsumexp1_i8(a, b) __float2half_rn(expf((float)(a)) + expf((float)(b)))
#define logsumexp1_i16(a, b) __float2half_rn(expf((float)(a)) + expf((float)(b)))
#define logsumexp1_i32(a, b) expf((float)(a)) + expf((float)(b))
#define logsumexp1_i64(a, b) exp((double)(a)) + exp((double)(b))
#define logsumexp1_f32(a, b) expf((a)) + expf((b))
#define logsumexp1_f64(a, b) exp((a)) + exp((b))

#define logsumexp1_u8(a, b) __float2half_rn(expf((float)(a)) + expf((float)(b)))
#define logsumexp1_u16(a, b) __float2half_rn(expf((float)(a)) + expf((float)(b)))
#define logsumexp1_u32(a, b) expf((float)(a)) + expf((float)(b))
#define logsumexp1_u64(a, b) exp((double)(a)) + exp((double)(b))

#define logsumexp1_f16(a, b) __float2half_rn(expf(__half2float(a)) + expf(__half2float(b)))
#define logsumexp1_bf16(a, b) __float2bfloat16_rn(expf(__bfloat162float(a)) + expf(__bfloat162float(b)))

#define logsumexp1_unary_bool(a) __float2half_rn(expf((float)(a)))
#define logsumexp1_unary_i8(a) __float2half_rn(expf((float)(a)))
#define logsumexp1_unary_i16(a) __float2half_rn(expf((float)(a)))
#define logsumexp1_unary_i32(a) expf((float)(a))
#define logsumexp1_unary_i64(a) exp((double)(a))
#define logsumexp1_unary_f32(a) expf(a)
#define logsumexp1_unary_f64(a) exp(a)

#define logsumexp1_unary_u8(a) __float2half_rn(expf((float)(a)))
#define logsumexp1_unary_u16(a) __float2half_rn(expf((float)(a)))
#define logsumexp1_unary_u32(a) expf((float)(a))
#define logsumexp1_unary_u64(a) exp((double)(a))

#define logsumexp1_unary_f16(a) __float2half_rn(expf(__half2float(a)))
#define logsumexp1_unary_bf16(a) __float2bfloat16_rn(expf(__bfloat162float(a)))

#define logsumexp2_bool(a, b) __hadd((a), (b))
#define logsumexp2_i8(a, b) __hadd((a), (b))
#define logsumexp2_i16(a, b) __hadd((a), (b))
#define logsumexp2_i32(a, b) (a) + (b)
#define logsumexp2_i64(a, b) (a) + (b)
#define logsumexp2_f32(a, b) (a) + (b)
#define logsumexp2_f64(a, b) (a) + (b)

#define logsumexp2_u8(a, b) __hadd((a), (b))
#define logsumexp2_u16(a, b) __hadd((a), (b))
#define logsumexp2_u32(a, b) (a) + (b)
#define logsumexp2_u64(a, b) (a) + (b)

#define logsumexp2_f16(a, b) __hadd((a), (b))
#define logsumexp2_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(i8, char, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(i16, short, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(i32, int, float, 0, logsumexp)
DEFINE_REDUCE_KERNEL(i64, long long, double, 0, logsumexp)

DEFINE_REDUCE_KERNEL(u8, unsigned char, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(u16, unsigned short, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(u32, unsigned int, float, 0, logsumexp)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, double, 0, logsumexp)

DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, logsumexp)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, logsumexp)

DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, logsumexp)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, logsumexp)