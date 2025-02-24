#pragma once

template <typename T, typename Func>
struct ProgressUpdater
{
    Func update_func;
    T *data;
    __device__ __forceinline__ ProgressUpdater(Func f, T *data) : update_func(f), data(data) {}
    __device__ __forceinline__ void update() { update_func(data); }
    __device__ __forceinline__ T get() const { return *data; }
    __device__ __forceinline__ void set_ptr(T *data) { this->data = data; }
};