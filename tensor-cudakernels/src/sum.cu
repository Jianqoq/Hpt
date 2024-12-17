#include <cuda_fp16.h>
#define WRAP 32

#define add_bool(a, b) ((unsigned char)a) + ((unsigned char)b)
#define add_i8(a, b) (a) + (b)
#define add_i16(a, b) (a) + (b)
#define add_i32(a, b) (a) + (b)
#define add_i64(a, b) (a) + (b)
#define add_f32(a, b) (a) + (b)
#define add_f64(a, b) (a) + (b)

#define add_u8(a, b) (a) + (b)
#define add_u16(a, b) (a) + (b)
#define add_u32(a, b) (a) + (b)
#define add_u64(a, b) (a) + (b)

#define add_f16(a, b) __half2float((a)) + __half2float((b))

#define cast_back_bool(a) ((bool)a)
#define cast_back_i8(a) (a)
#define cast_back_i16(a) (a)
#define cast_back_i32(a) (a)
#define cast_back_i64(a) (a)
#define cast_back_f32(a) (a)
#define cast_back_f64(a) (a)

#define cast_back_u8(a) (a)
#define cast_back_u16(a) (a)
#define cast_back_u32(a) (a)
#define cast_back_u64(a) (a)
#define cast_back_f16(a) __float2half((a))

#define atomicAdd_bool(a, b)                       \
    acquire_lock(&global_lock);                    \
    (a) = ((unsigned char)a) + ((unsigned char)b); \
    release_lock(&global_lock);

#define atomicAdd_i8(a, b)      \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_u8(a, b)      \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_i16(a, b)     \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_u16(a, b)     \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_i64(a, b)     \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_u64(a, b)     \
    acquire_lock(&global_lock); \
    (a) += (b);                 \
    release_lock(&global_lock);

#define atomicAdd_i32(a, b) atomicAdd(&a, b)
#define atomicAdd_u32(a, b) atomicAdd(&a, b)
#define atomicAdd_f32(a, b) atomicAdd(&a, b)
#define atomicAdd_f64(a, b) atomicAdd(&a, b)

#define atomicAdd_f16(a, b)     \
    acquire_lock(&global_lock); \
    (a) = add_f16(a, b);        \
    release_lock(&global_lock);

__device__ int global_lock = 0;

__device__ void acquire_lock(int *lock)
{
    while (atomicCAS(lock, 0, 1) != 0)
    {
    }
}

__device__ void release_lock(int *lock)
{
    atomicExch(lock, 0);
}

#define DEFINE_REDUCE_KERNEL(rust_type, type)                                                                                                                             \
    __device__ __forceinline__ void warpReduce_##rust_type(volatile type *sdata_##rust_type, unsigned int tid)                                                            \
    {                                                                                                                                                                     \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 32]));                                             \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 16]));                                             \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 8]));                                              \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 4]));                                              \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 2]));                                              \
        sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + 1]));                                              \
    }                                                                                                                                                                     \
    extern "C" __global__ void contiguous_fast_reduce_##rust_type(type *out, type *in, size_t size)                                                                       \
    {                                                                                                                                                                     \
        extern __shared__ type sdata_##rust_type[];                                                                                                                       \
        unsigned int tid = threadIdx.x;                                                                                                                                   \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                       \
        sdata_##rust_type[tid] = 0;                                                                                                                                       \
        if (i + blockDim.x < size)                                                                                                                                        \
        {                                                                                                                                                                 \
            sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(in[i], in[i + blockDim.x]));                                                                   \
        }                                                                                                                                                                 \
        else if (i < size)                                                                                                                                                \
        {                                                                                                                                                                 \
            sdata_##rust_type[tid] = in[i];                                                                                                                               \
        }                                                                                                                                                                 \
        __syncthreads();                                                                                                                                                  \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                          \
        {                                                                                                                                                                 \
            if (tid < s)                                                                                                                                                  \
            {                                                                                                                                                             \
                sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]));                                      \
            }                                                                                                                                                             \
            __syncthreads();                                                                                                                                              \
        }                                                                                                                                                                 \
        if (tid < WRAP)                                                                                                                                                   \
        {                                                                                                                                                                 \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                               \
        }                                                                                                                                                                 \
        if (tid == 0)                                                                                                                                                     \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                       \
    }                                                                                                                                                                     \
    extern "C" __global__ void uncontiguous_fast_reduce_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t size)                  \
    {                                                                                                                                                                     \
        extern __shared__ type sdata_##rust_type[];                                                                                                                       \
        unsigned int tid = threadIdx.x;                                                                                                                                   \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                       \
        sdata_##rust_type[tid] = 0;                                                                                                                                       \
        if (i + blockDim.x < size)                                                                                                                                        \
        {                                                                                                                                                                 \
            long long a_amount = i;                                                                                                                                       \
            long long b_amount = i + blockDim.x;                                                                                                                          \
            long long a_offset = 0;                                                                                                                                       \
            long long b_offset = 0;                                                                                                                                       \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                           \
            {                                                                                                                                                             \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                           \
                a_amount /= shape[j];                                                                                                                                     \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                           \
                b_amount /= shape[j];                                                                                                                                     \
            }                                                                                                                                                             \
            sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(in[a_offset], in[b_offset]));                                                                  \
        }                                                                                                                                                                 \
        else if (i < size)                                                                                                                                                \
        {                                                                                                                                                                 \
            long long a_amount = i;                                                                                                                                       \
            long long a_offset = 0;                                                                                                                                       \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                           \
            {                                                                                                                                                             \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                           \
                a_amount /= shape[j];                                                                                                                                     \
            }                                                                                                                                                             \
            sdata_##rust_type[tid] = in[a_offset];                                                                                                                        \
        }                                                                                                                                                                 \
        __syncthreads();                                                                                                                                                  \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                          \
        {                                                                                                                                                                 \
            if (tid < s)                                                                                                                                                  \
            {                                                                                                                                                             \
                sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]));                                      \
            }                                                                                                                                                             \
            __syncthreads();                                                                                                                                              \
        }                                                                                                                                                                 \
        if (tid < WRAP)                                                                                                                                                   \
        {                                                                                                                                                                 \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                               \
        }                                                                                                                                                                 \
        if (tid == 0)                                                                                                                                                     \
            out[blockIdx.x] = sdata_##rust_type[0];                                                                                                                       \
    }                                                                                                                                                                     \
    extern "C" __global__ void contiguous_reduce2_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t num_blocks_per_row)         \
    {                                                                                                                                                                     \
        extern __shared__ type sdata_##rust_type[];                                                                                                                       \
        unsigned int tid = threadIdx.x;                                                                                                                                   \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                       \
        sdata_##rust_type[tid] = 0;                                                                                                                                       \
        if (i + blockDim.x < cols)                                                                                                                                        \
        {                                                                                                                                                                 \
            long long a_offset = 0;                                                                                                                                       \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                   \
            long long b_offset = 0;                                                                                                                                       \
            long long b_amount = i + blockDim.x + blockIdx.y * cols;                                                                                                      \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                           \
            {                                                                                                                                                             \
                a_offset += (a_amount % shape[j]) * strides[j];                                                                                                           \
                a_amount /= shape[j];                                                                                                                                     \
                b_offset += (b_amount % shape[j]) * strides[j];                                                                                                           \
                b_amount /= shape[j];                                                                                                                                     \
            }                                                                                                                                                             \
            sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(in[a_offset], in[b_offset]));                                                                  \
        }                                                                                                                                                                 \
        else if (i < cols)                                                                                                                                                \
        {                                                                                                                                                                 \
            long long a_amount = i + blockIdx.y * cols;                                                                                                                   \
                                                                                                                                                                          \
            for (int j = ndim - 1; j >= 0; j--)                                                                                                                           \
            {                                                                                                                                                             \
                in += (a_amount % shape[j]) * strides[j];                                                                                                                 \
                a_amount /= shape[j];                                                                                                                                     \
            }                                                                                                                                                             \
            sdata_##rust_type[tid] = *in;                                                                                                                                 \
        }                                                                                                                                                                 \
        __syncthreads();                                                                                                                                                  \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                          \
        {                                                                                                                                                                 \
            if (tid < s)                                                                                                                                                  \
            {                                                                                                                                                             \
                sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]));                                      \
            }                                                                                                                                                             \
            __syncthreads();                                                                                                                                              \
        }                                                                                                                                                                 \
        if (tid < WRAP)                                                                                                                                                   \
        {                                                                                                                                                                 \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                               \
        }                                                                                                                                                                 \
        if (tid == 0)                                                                                                                                                     \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                     \
    }                                                                                                                                                                     \
                                                                                                                                                                          \
    extern "C" __global__ void contiguous_reduce22_##rust_type(type *out, type *in, size_t cols, size_t num_blocks_per_row)                                                           \
    {                                                                                                                                                                     \
        extern __shared__ type sdata_##rust_type[];                                                                                                                       \
        unsigned int tid = threadIdx.x;                                                                                                                                   \
        unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;                                                                                                       \
        sdata_##rust_type[tid] = 0;                                                                                                                                       \
        if (i + blockDim.x < cols)                                                                                                                                        \
        {                                                                                                                                                                 \
            sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(in[i + blockIdx.y * cols], in[i + blockDim.x + blockIdx.y * cols]));                           \
        }                                                                                                                                                                 \
        else if (i < cols)                                                                                                                                                \
        {                                                                                                                                                                 \
            sdata_##rust_type[tid] = in[i + blockIdx.y * cols];                                                                                                           \
        }                                                                                                                                                                 \
        __syncthreads();                                                                                                                                                  \
        for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)                                                                                                          \
        {                                                                                                                                                                 \
            if (tid < s)                                                                                                                                                  \
            {                                                                                                                                                             \
                sdata_##rust_type[tid] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[tid], sdata_##rust_type[tid + s]));                                      \
            }                                                                                                                                                             \
            __syncthreads();                                                                                                                                              \
        }                                                                                                                                                                 \
        if (tid < WRAP)                                                                                                                                                   \
        {                                                                                                                                                                 \
            warpReduce_##rust_type(sdata_##rust_type, tid);                                                                                                               \
        }                                                                                                                                                                 \
        if (tid == 0)                                                                                                                                                     \
            out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata_##rust_type[0];                                                                                     \
    }                                                                                                                                                                     \
    extern "C" __global__ void contiguous_reduce3_##rust_type(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)           \
    {                                                                                                                                                                     \
        extern __shared__ type sdata_##rust_type[];                                                                                                                       \
        unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;                                                                                                        \
        unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                                     \
        unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;                                                                                                     \
        sdata_##rust_type[tid] = 0;                                                                                                                                       \
        if (col_idx >= cols || row_idx >= rows)                                                                                                                           \
        {                                                                                                                                                                 \
            return;                                                                                                                                                       \
        }                                                                                                                                                                 \
        unsigned int idx = row_idx * cols + col_idx;                                                                                                                      \
        long long offset = 0;                                                                                                                                             \
        for (int j = ndim - 1; j >= 0; j--)                                                                                                                               \
        {                                                                                                                                                                 \
            offset += (idx % shape[j]) * strides[j];                                                                                                                      \
            idx /= shape[j];                                                                                                                                              \
        }                                                                                                                                                                 \
        sdata_##rust_type[tid] = in[offset];                                                                                                                              \
        __syncthreads();                                                                                                                                                  \
        if (threadIdx.y == 0)                                                                                                                                             \
        {                                                                                                                                                                 \
            for (unsigned int s = 1; s < blockDim.y; s++)                                                                                                                 \
            {                                                                                                                                                             \
                sdata_##rust_type[threadIdx.x] = cast_back_##rust_type(add_##rust_type(sdata_##rust_type[threadIdx.x], sdata_##rust_type[s * blockDim.x + threadIdx.x])); \
            }                                                                                                                                                             \
            atomicAdd_##rust_type(out[col_idx], sdata_##rust_type[threadIdx.x]);                                                                                          \
        }                                                                                                                                                                 \
    }

DEFINE_REDUCE_KERNEL(bool, bool)
DEFINE_REDUCE_KERNEL(i8, char)
DEFINE_REDUCE_KERNEL(i16, short)
DEFINE_REDUCE_KERNEL(i32, int)
DEFINE_REDUCE_KERNEL(i64, long long)
DEFINE_REDUCE_KERNEL(u8, unsigned char)
DEFINE_REDUCE_KERNEL(u16, unsigned short)
DEFINE_REDUCE_KERNEL(u32, unsigned int)
DEFINE_REDUCE_KERNEL(u64, unsigned long long)
DEFINE_REDUCE_KERNEL(f32, float)
DEFINE_REDUCE_KERNEL(f64, double)
DEFINE_REDUCE_KERNEL(f16, __half)
