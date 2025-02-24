#pragma once

#include "fast_divmod.cuh"

template <typename T>
class IndexCalculator
{
public:
    __device__ __forceinline__ T get(long long idx) const { return T(); }
    __device__ __forceinline__ T *get_ptr(long long idx) const { return nullptr; }
};

template <typename T>
struct ContiguousIndexCalculator : public IndexCalculator<T>
{
    T *data;
    __device__ __forceinline__ T get(long long idx) const
    {
        return data[idx];
    }
    __device__ __forceinline__ T *get_ptr(long long idx) const { return &data[idx]; }
    __device__ __forceinline__ ContiguousIndexCalculator(T *data) : data(data) {}
};

template <typename T>
struct UncontiguousIndexCalculator : public IndexCalculator<T>
{
    T *data;
    long long *shape;
    long long *strides;
    int ndim;
    __device__ __forceinline__ T get(long long idx) const
    {
        long long r;
        long long offset = 0;
        for (int j = ndim - 1; j >= 0; j--)
        {
            divmod(idx, shape[j], idx, r);
            offset += r * strides[j];
        }
        return data[offset];
    }
    __device__ __forceinline__ T *get_ptr(long long idx) const
    {
        long long r;
        long long offset = 0;
        for (int j = ndim - 1; j >= 0; j--)
        {
            divmod(idx, shape[j], idx, r);
            offset += r * strides[j];
        }
        return &data[offset];
    }
    __device__ __forceinline__ UncontiguousIndexCalculator(T *data, long long *shape, long long *strides, int ndim) : data(data), shape(shape), strides(strides), ndim(ndim) {}
};
