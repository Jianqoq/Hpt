#include <stdint.h>
#include "../utils/type_alias.cuh"
#include "../utils/type_cast.cuh"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

template <typename T, typename Intermediate>
__device__ __forceinline__ void layernorm_post(T *input, T *gamma, T *beta, size_t size, size_t channels)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
    {
        int c = idx % channels;
        Intermediate gamma_casted = cast<T, Intermediate>(gamma[c]);
        Intermediate beta_casted = cast<T, Intermediate>(beta[c]);
        Intermediate input_casted = cast<T, Intermediate>(input[idx]);
        input[idx] = cast<Intermediate, T>(input_casted * gamma_casted + beta_casted);
    }
}

#define DEFINE_LAYERNORM_POST_KERNEL(func_name, type, intermediate)                                          \
    extern "C" __global__ void func_name(type *input, type *gamma, type *beta, size_t size, size_t channels) \
    {                                                                                                        \
        layernorm_post<type, intermediate>(input, gamma, beta, size, channels);                              \
    }

DEFINE_LAYERNORM_POST_KERNEL(layernorm_post_f32, f32, f32);
DEFINE_LAYERNORM_POST_KERNEL(layernorm_post_f64, f64, f64);

DEFINE_LAYERNORM_POST_KERNEL(layernorm_post_f16, f16, f32);
DEFINE_LAYERNORM_POST_KERNEL(layernorm_post_bf16, bf16, f32);
