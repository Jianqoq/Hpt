#define type float
#define WRAP 32

__device__ __forceinline__ void warpReduce(volatile type *sdata, unsigned int tid)
{
#pragma unroll
    for (int offset = WRAP; offset > 0; offset >>= 1)
    {
        sdata[tid] += sdata[tid + offset];
    }
}

extern "C" __global__ void contiguous_reduce(type *out, type *in, size_t size)
{
    extern __shared__ type sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = 0;

    while (i < size)
    {
        if (i + blockDim.x < size)
        {
            sdata[tid] += in[i] + in[i + blockDim.x];
        }
        else if (i < size)
        {
            sdata[tid] += in[i];
        }
        i += blockDim.x * gridDim.x;
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
