#include <stdint.h>
#include "../utils/type_alias.cuh"
#include "../utils/type_cast.cuh"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

template <typename T, typename Intermediate>
__device__ __forceinline__ void batchnorm_forward(T *input, T *gamma, T *beta, T *mean, T *var, T eps, size_t size, size_t channels)
{
    Intermediate eps_casted = cast<T, Intermediate>(eps);
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)
    {
        int c = idx % channels;
        Intermediate var_casted = cast<T, Intermediate>(var[c]);
        Intermediate mean_casted = cast<T, Intermediate>(mean[c]);
        Intermediate gamma_casted = cast<T, Intermediate>(gamma[c]);
        Intermediate beta_casted = cast<T, Intermediate>(beta[c]);
        Intermediate input_casted = cast<T, Intermediate>(input[idx]);
        input[idx] = cast<Intermediate, T>((input_casted - mean_casted) / sqrt(var_casted + eps_casted) * gamma_casted + beta_casted);
    }
}

#define DEFINE_BATCHNORM_FORWARD_KERNEL(func_name, type, intermediate)                                                                        \
    extern "C" __global__ void func_name(type *input, type *gamma, type *beta, type *mean, type *var, type eps, size_t size, size_t channels) \
    {                                                                                                                                         \
        batchnorm_forward<type, intermediate>(input, gamma, beta, mean, var, eps, size, channels);                                            \
    }

DEFINE_BATCHNORM_FORWARD_KERNEL(batchnorm_forward_f32, f32, f32);
DEFINE_BATCHNORM_FORWARD_KERNEL(batchnorm_forward_f64, f64, f64);

DEFINE_BATCHNORM_FORWARD_KERNEL(batchnorm_forward_f16, f16, f32);
DEFINE_BATCHNORM_FORWARD_KERNEL(batchnorm_forward_bf16, bf16, f32);
