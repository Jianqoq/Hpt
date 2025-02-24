#pragma once
template <typename T>
__device__ __forceinline__ T get_element(T *in, long long *shape, long long *strides, size_t ndim, long long idx)
{
    for (int i = ndim - 1; i >= 0; i--)
    {
        long long r;
        divmod(idx, shape[i], idx, r);
        in += r * strides[i];
    }
    return *in;
}

template <typename T>
__device__ __forceinline__ void divmod(T a, T b, T &div, T &mod)
{
    div = a / b;
    mod = a - div * b;
}
