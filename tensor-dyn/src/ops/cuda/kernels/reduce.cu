#define type float
#define WRAP 32

__device__ __forceinline__ void warpReduce(volatile type *sdata, unsigned int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

extern "C" __global__ void contiguous_reduce(type *out, type *in, size_t size)
{
    extern __shared__ type sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i + blockDim.x < size)
    {
        sdata[tid] += in[i] + in[i + blockDim.x];
    }
    else if (i < size)
    {
        sdata[tid] += in[i];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < WRAP)
    {
        warpReduce(sdata, tid);
    }
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

extern "C" __global__ void contiguous_reduce2(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t num_blocks_per_row)
{
    extern __shared__ type sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i + blockDim.x < cols)
    {
        long long a_offset = 0;
        long long a_amount = i + blockIdx.y * cols;
        long long b_offset = 0;
        long long b_amount = i + blockDim.x + blockIdx.y * cols;

        for (int j = ndim - 1; j >= 0; j--)
        {
            a_offset += (a_amount % shape[j]) * strides[j];
            a_amount /= shape[j];
            b_offset += (b_amount % shape[j]) * strides[j];
            b_amount /= shape[j];
        }

        sdata[tid] += in[a_offset] + in[b_offset];
    }
    else if (i < cols)
    {
        long long a_amount = i + blockIdx.y * cols;

        for (int j = ndim - 1; j >= 0; j--)
        {
            in += (a_amount % shape[j]) * strides[j];
            a_amount /= shape[j];
        }
        sdata[tid] += *in;
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < WRAP)
    {
        warpReduce(sdata, tid);
    }
    if (tid == 0)
        out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata[0];
}

extern "C" __global__ void contiguous_reduce22(type *out, type *in, size_t cols, size_t num_blocks_per_row)
{
    extern __shared__ type sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i + blockDim.x < cols)
    {
        sdata[tid] += in[i + blockIdx.y * cols] + in[i + blockDim.x + blockIdx.y * cols];
    }
    else if (i < cols)
    {
        sdata[tid] += in[i + blockIdx.y * cols];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > WRAP; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < WRAP)
    {
        warpReduce(sdata, tid);
    }
    if (tid == 0)
        out[blockIdx.x + blockIdx.y * num_blocks_per_row] = sdata[0];
}

extern "C" __global__ void contiguous_reduce3(type *out, type *in, long long *shape, long long *strides, size_t ndim, size_t cols, size_t rows)
{
    extern __shared__ type sdata[];
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    sdata[tid] = 0;
    if (col_idx >= cols || row_idx >= rows)
    {
        return;
    }
    unsigned int idx = row_idx * cols + col_idx;
    long long offset = 0;
    for (int j = ndim - 1; j >= 0; j--)
    {
        offset += (idx % shape[j]) * strides[j];
        idx /= shape[j];
    }
    sdata[tid] = in[offset];
    __syncthreads();
    if (threadIdx.y == 0)
    {
        for (unsigned int s = 1; s < blockDim.y; s++)
        {
            sdata[threadIdx.x] += sdata[s * blockDim.x + threadIdx.x];
        }
        atomicAdd(&out[col_idx], sdata[threadIdx.x]);
    }
}
