#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct TypeUtils
{
    __device__ __forceinline__ static T limit_max()
    {
#define LIMIT_MAX(type, max)               \
    if constexpr (std::is_same_v<T, type>) \
    {                                      \
        return max;                        \
    }
        LIMIT_MAX(bool, true);
        LIMIT_MAX(int8_t, INT8_MAX);
        LIMIT_MAX(int16_t, INT16_MAX);
        LIMIT_MAX(int32_t, INT32_MAX);
        LIMIT_MAX(int64_t, INT64_MAX);
        LIMIT_MAX(uint8_t, UINT8_MAX);
        LIMIT_MAX(uint16_t, UINT16_MAX);
        LIMIT_MAX(uint32_t, UINT32_MAX);
        LIMIT_MAX(uint64_t, UINT64_MAX);
        LIMIT_MAX(float, INFINITY);
        LIMIT_MAX(double, INFINITY);
        LIMIT_MAX(__half, __half((unsigned short)31744));
        LIMIT_MAX(__nv_bfloat16, __nv_bfloat16((unsigned short)0x7F80));
    }
    __device__ __forceinline__ static T limit_min()
    {
#define LIMIT_MIN(type, min)               \
    if constexpr (std::is_same_v<T, type>) \
    {                                      \
        return min;                        \
    }
        LIMIT_MIN(bool, false);
        LIMIT_MIN(int8_t, INT8_MIN);
        LIMIT_MIN(int16_t, INT16_MIN);
        LIMIT_MIN(int32_t, INT32_MIN);
        LIMIT_MIN(int64_t, INT64_MIN);
        LIMIT_MIN(uint8_t, 0);
        LIMIT_MIN(uint16_t, 0);
        LIMIT_MIN(uint32_t, 0);
        LIMIT_MIN(uint64_t, 0);
        LIMIT_MIN(float, -INFINITY);
        LIMIT_MIN(double, -INFINITY);
        LIMIT_MIN(__half, __half((unsigned short)0xFC00U));
        LIMIT_MIN(__nv_bfloat16, __nv_bfloat16((unsigned short)0xFF80U));
    }
};
