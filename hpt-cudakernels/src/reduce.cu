#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define WRAP 32

#define DEFINE_REDUCE_KERNEL(rust_type, in_type, out_type, INIT_VAL, METHOD)                                                                                                             \
    __device__ __forceinline__ void warpReduce_##METHOD##_##rust_type(volatile out_type *METHOD##sdata_##rust_type, unsigned int tid)                                                    \
    {                                                                                                                                                                                    \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP]);                                                    \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 2]);                                                \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 4]);                                                \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 8]);                                                \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 16]);                                               \
        METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + WRAP / 32]);                                               \
    }                                                                                                                                                                                    \
    extern "C" __global__ void contiguous_##METHOD##_##rust_type(out_type *out, in_type *in, size_t size)                                                                                \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.x;                                                                                                                                                  \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                      \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (i + blockDim.x < size)                                                                                                                                                       \
        {                                                                                                                                                                                \
            METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(in[i], in[i + blockDim.x]);                                                                                            \
        }                                                                                                                                                                                \
        else if (i < size)                                                                                                                                                               \
        {                                                                                                                                                                                \
            METHOD##sdata_##rust_type[tid] = in[i];                                                                                                                                      \
        }                                                                                                                                                                                \
        __syncthreads();                                                                                                                                                                 \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                         \
        {                                                                                                                                                                                \
            if (tid < s)                                                                                                                                                                 \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                               \
            }                                                                                                                                                                            \
            __syncthreads();                                                                                                                                                             \
        }                                                                                                                                                                                \
        if (tid < WRAP)                                                                                                                                                                  \
        {                                                                                                                                                                                \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                           \
        }                                                                                                                                                                                \
        if (tid == 0)                                                                                                                                                                    \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                              \
    }                                                                                                                                                                                    \
    extern "C" __global__ void uncontiguous_##METHOD##_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t size)                           \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.x;                                                                                                                                                  \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                      \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (i + blockDim.x < size)                                                                                                                                                       \
        {                                                                                                                                                                                \
            long long a_amount = i;                                                                                                                                                      \
            long long b_amount = i + blockDim.x;                                                                                                                                         \
            long long a_offset = 0;                                                                                                                                                      \
            long long b_offset = 0;                                                                                                                                                      \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                          \
            {                                                                                                                                                                            \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                          \
                a_amount /= shape[j];                                                                                                                                                    \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                          \
                b_amount /= shape[j];                                                                                                                                                    \
            }                                                                                                                                                                            \
            METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(in[a_offset], in[b_offset]);                                                                                           \
        }                                                                                                                                                                                \
        else if (i < size)                                                                                                                                                               \
        {                                                                                                                                                                                \
            long long a_amount = i;                                                                                                                                                      \
            long long a_offset = 0;                                                                                                                                                      \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                          \
            {                                                                                                                                                                            \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                          \
                a_amount /= shape[j];                                                                                                                                                    \
            }                                                                                                                                                                            \
            METHOD##sdata_##rust_type[tid] = in[a_offset];                                                                                                                               \
        }                                                                                                                                                                                \
        __syncthreads();                                                                                                                                                                 \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                         \
        {                                                                                                                                                                                \
            if (tid < s)                                                                                                                                                                 \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                               \
            }                                                                                                                                                                            \
            __syncthreads();                                                                                                                                                             \
        }                                                                                                                                                                                \
        if (tid < WRAP)                                                                                                                                                                  \
        {                                                                                                                                                                                \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                           \
        }                                                                                                                                                                                \
        if (tid == 0)                                                                                                                                                                    \
            out[blockIdx.x] = METHOD##sdata_##rust_type[0];                                                                                                                              \
    }                                                                                                                                                                                    \
    extern "C" __global__ void contiguous_##METHOD##2_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t num_blocks_per_row) \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.x;                                                                                                                                                  \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                      \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (i + blockDim.x < cols)                                                                                                                                                       \
        {                                                                                                                                                                                \
            long long a_offset = 0;                                                                                                                                                      \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                                  \
            long long b_offset = 0;                                                                                                                                                      \
            long long b_amount = i + blockDim.x + blockIdx.y * cols;                                                                                                                     \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                          \
            {                                                                                                                                                                            \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                                          \
                a_amount /= shape[j];                                                                                                                                                    \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                                          \
                b_amount /= shape[j];                                                                                                                                                    \
            }                                                                                                                                                                            \
            METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(in[a_offset], in[b_offset]);                                                                                           \
        }                                                                                                                                                                                \
        else if (i < cols)                                                                                                                                                               \
        {                                                                                                                                                                                \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                                  \
                                                                                                                                                                                         \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                                          \
            {                                                                                                                                                                            \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                                \
                a_amount /= shape[j];                                                                                                                                                    \
            }                                                                                                                                                                            \
            METHOD##sdata_##rust_type[tid] = *in;                                                                                                                                        \
        }                                                                                                                                                                                \
        __syncthreads();                                                                                                                                                                 \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                         \
        {                                                                                                                                                                                \
            if (tid < s)                                                                                                                                                                 \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                               \
            }                                                                                                                                                                            \
            __syncthreads();                                                                                                                                                             \
        }                                                                                                                                                                                \
        if (tid < WRAP)                                                                                                                                                                  \
        {                                                                                                                                                                                \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                           \
        }                                                                                                                                                                                \
        if (tid == 0)                                                                                                                                                                    \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = METHOD##sdata_##rust_type[0];                                                                                            \
    }                                                                                                                                                                                    \
                                                                                                                                                                                         \
    extern "C" __global__ void contiguous_##METHOD##22_##rust_type(out_type *out, in_type *in, size_t cols, size_t num_blocks_per_row)                                                   \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.x;                                                                                                                                                  \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                      \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (i + blockDim.x < cols)                                                                                                                                                       \
        {                                                                                                                                                                                \
            METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(in[i + blockIdx.y * cols], in[i + blockDim.x + blockIdx.y * cols]);                                                    \
        }                                                                                                                                                                                \
        else if (i < cols)                                                                                                                                                               \
        {                                                                                                                                                                                \
            METHOD##sdata_##rust_type[tid] = in[i + blockIdx.y * cols];                                                                                                                  \
        }                                                                                                                                                                                \
        __syncthreads();                                                                                                                                                                 \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                                         \
        {                                                                                                                                                                                \
            if (tid < s)                                                                                                                                                                 \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[tid] = METHOD##_##rust_type(METHOD##sdata_##rust_type[tid], METHOD##sdata_##rust_type[tid + s]);                                               \
            }                                                                                                                                                                            \
            __syncthreads();                                                                                                                                                             \
        }                                                                                                                                                                                \
        if (tid < WRAP)                                                                                                                                                                  \
        {                                                                                                                                                                                \
            warpReduce_##METHOD##_##rust_type(METHOD##sdata_##rust_type, tid);                                                                                                           \
        }                                                                                                                                                                                \
        if (tid == 0)                                                                                                                                                                    \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = METHOD##sdata_##rust_type[0];                                                                                            \
    }                                                                                                                                                                                    \
    extern "C" __global__ void contiguous_##METHOD##3_##rust_type(out_type *out, in_type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)               \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                       \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                    \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                    \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                          \
        {                                                                                                                                                                                \
            return;                                                                                                                                                                      \
        }                                                                                                                                                                                \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                     \
        long long offset = 0;                                                                                                                                                            \
        for (int j = ndim - 1; j >= 0; j--)                                                                                                                                              \
        {                                                                                                                                                                                \
            offset += (idx % shape[j]) * strides[j];                                                                                                                                     \
            idx /= shape[j];                                                                                                                                                             \
        }                                                                                                                                                                                \
        METHOD##sdata_##rust_type[tid] = in[offset];                                                                                                                                     \
        __syncthreads();                                                                                                                                                                 \
        if (threadIdx.y == 0)                                                                                                                                                            \
        {                                                                                                                                                                                \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[threadIdx.x] = METHOD##_##rust_type(METHOD##sdata_##rust_type[threadIdx.x], METHOD##sdata_##rust_type[s * blockDim.x + threadIdx.x]);          \
            }                                                                                                                                                                            \
            out[col_idx + blockIdx.y * cols] = METHOD##sdata_##rust_type[threadIdx.x];                                                                                                   \
        }                                                                                                                                                                                \
    }                                                                                                                                                                                    \
    extern "C" __global__ void contiguous_##METHOD##33_##rust_type(out_type *out, in_type *in, size_t ndim, size_t cols, size_t rows)                                                    \
    {                                                                                                                                                                                    \
        extern __shared__ out_type METHOD##sdata_##rust_type[];                                                                                                                          \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                                       \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                                    \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                                    \
        METHOD##sdata_##rust_type[tid] = INIT_VAL;                                                                                                                                       \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                                          \
        {                                                                                                                                                                                \
            return;                                                                                                                                                                      \
        }                                                                                                                                                                                \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                                     \
        METHOD##sdata_##rust_type[tid] = in[idx];                                                                                                                                        \
        __syncthreads();                                                                                                                                                                 \
        if (threadIdx.y == 0)                                                                                                                                                            \
        {                                                                                                                                                                                \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                                \
            {                                                                                                                                                                            \
                METHOD##sdata_##rust_type[threadIdx.x] = METHOD##_##rust_type(METHOD##sdata_##rust_type[threadIdx.x], METHOD##sdata_##rust_type[s * blockDim.x + threadIdx.x]);          \
            }                                                                                                                                                                            \
            out[col_idx + blockIdx.y * cols] = METHOD##sdata_##rust_type[threadIdx.x];                                                                                                   \
        }                                                                                                                                                                                \
    }

// prod
#define prod_bool(a, b) ((bool)((unsigned char)a) * ((unsigned char)b))
#define prod_i8(a, b) (a) * (b)
#define prod_i16(a, b) (a) * (b)
#define prod_i32(a, b) (a) * (b)
#define prod_i64(a, b) (a) * (b)
#define prod_f32(a, b) (a) * (b)
#define prod_f64(a, b) (a) * (b)

#define prod_u8(a, b) (a) * (b)
#define prod_u16(a, b) (a) * (b)
#define prod_u32(a, b) (a) * (b)
#define prod_u64(a, b) (a) * (b)

#define prod_f16(a, b) __hmul((a), (b))
#define prod_bf16(a, b) __hmul((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, bool, 1, prod)
DEFINE_REDUCE_KERNEL(i8, char, char, 1, prod)
DEFINE_REDUCE_KERNEL(i16, short, short, 1, prod)
DEFINE_REDUCE_KERNEL(i32, int, int, 1, prod)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 1, prod)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 1, prod)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 1, prod)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 1, prod)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 1, prod)
DEFINE_REDUCE_KERNEL(f32, float, float, 1.0f, prod)
DEFINE_REDUCE_KERNEL(f64, double, double, 1.0, prod)
DEFINE_REDUCE_KERNEL(f16, __half, __half, __half((unsigned short)0x3C00U), prod)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16((unsigned short)0x3F80U), prod)

// nanprod
#define nanprod_bool(a, b) ((bool)((unsigned char)a) * ((unsigned char)b))
#define nanprod_i8(a, b) (a) * (b)
#define nanprod_i16(a, b) (a) * (b)
#define nanprod_i32(a, b) (a) * (b)
#define nanprod_i64(a, b) (a) * (b)
#define nanprod_f32(a, b) (isnan(a) ? (isnan(b) ? 1.0f : b) : (isnan(b) ? a : a * b))
#define nanprod_f64(a, b) (isnan(a) ? (isnan(b) ? 1.0 : b) : (isnan(b) ? a : a * b))

#define nanprod_u8(a, b) (a) * (b)
#define nanprod_u16(a, b) (a) * (b)
#define nanprod_u32(a, b) (a) * (b)
#define nanprod_u64(a, b) (a) * (b)

#define nanprod_f16(a, b) __hmul((a), (b))
#define nanprod_bf16(a, b) __hmul((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, bool, 1, nanprod)
DEFINE_REDUCE_KERNEL(i8, char, char, 1, nanprod)
DEFINE_REDUCE_KERNEL(i16, short, short, 1, nanprod)
DEFINE_REDUCE_KERNEL(i32, int, int, 1, nanprod)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 1, nanprod)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 1, nanprod)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 1, nanprod)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 1, nanprod)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 1, nanprod)
DEFINE_REDUCE_KERNEL(f32, float, float, 1.0f, nanprod)
DEFINE_REDUCE_KERNEL(f64, double, double, 1.0, nanprod)
DEFINE_REDUCE_KERNEL(f16, __half, __half, __half((unsigned short)0x3C00U), nanprod)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16((unsigned short)0x3F80U), nanprod)

// sum
#define sum_bool(a, b) ((bool)((unsigned char)a) + ((unsigned char)b))
#define sum_i8(a, b) (a) + (b)
#define sum_i16(a, b) (a) + (b)
#define sum_i32(a, b) (a) + (b)
#define sum_i64(a, b) (a) + (b)
#define sum_f32(a, b) (a) + (b)
#define sum_f64(a, b) (a) + (b)

#define sum_u8(a, b) (a) + (b)
#define sum_u16(a, b) (a) + (b)
#define sum_u32(a, b) (a) + (b)
#define sum_u64(a, b) (a) + (b)

#define sum_f16(a, b) __hadd((a), (b))
#define sum_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, bool, false, sum)
DEFINE_REDUCE_KERNEL(i8, char, char, 0, sum)
DEFINE_REDUCE_KERNEL(i16, short, short, 0, sum)
DEFINE_REDUCE_KERNEL(i32, int, int, 0, sum)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 0, sum)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 0, sum)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 0, sum)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 0, sum)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 0, sum)
DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, sum)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, sum)
DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, sum)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, sum)

// nansum
#define nansum_bool(a, b) ((bool)((unsigned char)a) + ((unsigned char)b))
#define nansum_i8(a, b) (a) + (b)
#define nansum_i16(a, b) (a) + (b)
#define nansum_i32(a, b) (a) + (b)
#define nansum_i64(a, b) (a) + (b)
#define nansum_f32(a, b) (isnan(a) ? (isnan(b) ? 0.0f : b) : (isnan(b) ? a : a + b))

#define nansum_f64(a, b) (isnan(a) ? (isnan(b) ? 0.0 : b) : (isnan(b) ? a : a + b))

#define nansum_u8(a, b) (a) + (b)
#define nansum_u16(a, b) (a) + (b)
#define nansum_u32(a, b) (a) + (b)
#define nansum_u64(a, b) (a) + (b)

#define nansum_f16(a, b) __hadd((a), (b))
#define nansum_bf16(a, b) __hadd((a), (b))

DEFINE_REDUCE_KERNEL(bool, bool, bool, false, nansum)
DEFINE_REDUCE_KERNEL(i8, char, char, 0, nansum)
DEFINE_REDUCE_KERNEL(i16, short, short, 0, nansum)
DEFINE_REDUCE_KERNEL(i32, int, int, 0, nansum)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 0, nansum)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 0, nansum)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 0, nansum)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 0, nansum)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 0, nansum)
DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, nansum)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, nansum)
DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, nansum)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, nansum)

// min
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

DEFINE_REDUCE_KERNEL(bool, bool, bool, 1, min)
DEFINE_REDUCE_KERNEL(i8, char, char, SCHAR_MAX, min)
DEFINE_REDUCE_KERNEL(i16, short, short, SHRT_MAX, min)
DEFINE_REDUCE_KERNEL(i32, int, int, INT_MAX, min)
DEFINE_REDUCE_KERNEL(i64, long long, long long, LLONG_MAX, min)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, UCHAR_MAX, min)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, USHRT_MAX, min)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, UINT_MAX, min)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, ULONG_MAX, min)
DEFINE_REDUCE_KERNEL(f32, float, float, INFINITY, min)
DEFINE_REDUCE_KERNEL(f64, double, double, INFINITY, min)
DEFINE_REDUCE_KERNEL(f16, __half, __half, __half((unsigned short)31744), min)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16((unsigned short)0x7F80), min)

// max
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

DEFINE_REDUCE_KERNEL(bool, bool, bool, 0, max)
DEFINE_REDUCE_KERNEL(i8, char, char, SCHAR_MIN, max)
DEFINE_REDUCE_KERNEL(i16, short, short, SHRT_MIN, max)
DEFINE_REDUCE_KERNEL(i32, int, int, INT_MIN, max)
DEFINE_REDUCE_KERNEL(i64, long long, long long, LLONG_MIN, max)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 0, max)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 0, max)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 0, max)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 0, max)
DEFINE_REDUCE_KERNEL(f32, float, float, -INFINITY, max)
DEFINE_REDUCE_KERNEL(f64, double, double, -INFINITY, max)
DEFINE_REDUCE_KERNEL(f16, __half, __half, __half((unsigned short)0xFC00U), max)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16((unsigned short)0xFF80U), max)

// all
#define all_bool(a, b) ((bool)a) && ((bool)b)
#define all_i8(a, b) ((bool)a) && ((bool)b)
#define all_i16(a, b) ((bool)a) && ((bool)b)
#define all_i32(a, b) ((bool)a) && ((bool)b)
#define all_i64(a, b) ((bool)a) && ((bool)b)
#define all_f32(a, b) ((bool)a) && ((bool)b)
#define all_f64(a, b) ((bool)a) && ((bool)b)

#define all_u8(a, b) ((bool)a) && ((bool)b)
#define all_u16(a, b) ((bool)a) && ((bool)b)
#define all_u32(a, b) ((bool)a) && ((bool)b)
#define all_u64(a, b) ((bool)a) && ((bool)b)

#define all_f16(a, b) (a && b)
#define all_bf16(a, b) (a && b)

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

// any
#define any_bool(a, b) (a) || (b)
#define any_i8(a, b) (a) || (b)
#define any_i16(a, b) (a) || (b)
#define any_i32(a, b) (a) || (b)
#define any_i64(a, b) (a) || (b)
#define any_f32(a, b) (a) || (b)
#define any_f64(a, b) (a) || (b)

#define any_u8(a, b) (a) || (b)
#define any_u16(a, b) (a) || (b)
#define any_u32(a, b) (a) || (b)
#define any_u64(a, b) (a) || (b)

#define any_f16(a, b) (a) || (b)
#define any_bf16(a, b) (a) || (b)

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

// reducel1
#define reducel1_bool(a, b) ((bool)((unsigned char)a) + ((unsigned char)b))
#define reducel1_i8(a, b) (abs(a)) + (abs(b))
#define reducel1_i16(a, b) (abs(a)) + (abs(b))
#define reducel1_i32(a, b) (abs(a)) + (abs(b))
#define reducel1_i64(a, b) (abs(a)) + (abs(b))
#define reducel1_f32(a, b) (abs(a)) + (abs(b))
#define reducel1_f64(a, b) (abs(a)) + (abs(b))

#define reducel1_u8(a, b) (a) + (b)
#define reducel1_u16(a, b) (a) + (b)
#define reducel1_u32(a, b) (a) + (b)
#define reducel1_u64(a, b) (a) + (b)

#define reducel1_f16(a, b) __hadd(__habs((a)), __habs((b)))
#define reducel1_bf16(a, b) __hadd(__habs((a)), __habs((b)))

DEFINE_REDUCE_KERNEL(bool, bool, bool, 0, reducel1)
DEFINE_REDUCE_KERNEL(i8, char, char, 0, reducel1)
DEFINE_REDUCE_KERNEL(i16, short, short, 0, reducel1)
DEFINE_REDUCE_KERNEL(i32, int, int, 0, reducel1)
DEFINE_REDUCE_KERNEL(i64, long long, long long, 0, reducel1)
DEFINE_REDUCE_KERNEL(u8, unsigned char, unsigned char, 0, reducel1)
DEFINE_REDUCE_KERNEL(u16, unsigned short, unsigned short, 0, reducel1)
DEFINE_REDUCE_KERNEL(u32, unsigned int, unsigned int, 0, reducel1)
DEFINE_REDUCE_KERNEL(u64, unsigned long long, unsigned long long, 0, reducel1)
DEFINE_REDUCE_KERNEL(f32, float, float, 0.0f, reducel1)
DEFINE_REDUCE_KERNEL(f64, double, double, 0.0, reducel1)
DEFINE_REDUCE_KERNEL(f16, __half, __half, 0, reducel1)
DEFINE_REDUCE_KERNEL(bf16, __nv_bfloat16, __nv_bfloat16, 0, reducel1)
