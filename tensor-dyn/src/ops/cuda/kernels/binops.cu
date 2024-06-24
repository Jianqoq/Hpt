#include "binops.cuh"

extern "C" __global__ void add_float(float *a, float *b, float *c, size_t size, size_t* shape, size_t* strides)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
}