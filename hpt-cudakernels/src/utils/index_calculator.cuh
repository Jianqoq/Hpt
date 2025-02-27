#pragma once

#include "fast_divmod.cuh"

template <typename T>
class IndexCalculator
{
public:
    __device__ __forceinline__ T get(long long idx) const { return T(); }
    __device__ __forceinline__ T *get_ptr(long long idx) const { return nullptr; }
    __device__ __forceinline__ void cal_coord(long long idx, long long *coord)
    {
        for (int j = ndim - 1; j >= 0; j--)
        {
            divmod(idx, shape[j], idx, r);
            coord[j] = r;
        }
    }
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

template <typename T>
struct FastUncontiguousIndexCalculator : public IndexCalculator<T>
{
    T *data;
    FastDivmod *shape;
    int *strides;
    int ndim;
    __device__ __forceinline__ T get(int idx) const
    {
        T *res = data;
        for (int j = ndim - 1; j >= 0; j--)
        {
            int remainder;
            shape[j].fast_divmod(idx, remainder, idx);
            res += remainder * strides[j];
        }
        return *res;
    }
    __device__ __forceinline__ T *get_ptr(int idx) const
    {
        T *res = data;
        for (int j = ndim - 1; j >= 0; j--)
        {
            int remainder;
            shape[j].fast_divmod(idx, remainder, idx);
            res += remainder * strides[j];
        }
        return res;
    }
    __device__ __forceinline__ void cal_coord(int idx, int *coord)
    {
        for (int j = ndim - 1; j >= 0; j--)
        {
            int remainder;
            shape[j].fast_divmod(idx, remainder, idx);
            coord[j] = remainder;
        }
    }
    __device__ __forceinline__ FastUncontiguousIndexCalculator(T *data, FastDivmod *shape, int *strides, int ndim) : data(data), shape(shape), strides(strides), ndim(ndim) {}
};
