#pragma once
#include "type_alias.cuh"
#include "vec_types.cuh"

template <typename T>
struct VecMaker
{
    using Vec2 = typename VectorTrait<T, 2>::type;
    using Vec3 = typename VectorTrait<T, 3>::type;
    using Vec4 = typename VectorTrait<T, 4>::type;

    static __device__ __forceinline__ Vec2 make(T value, T value2);
    static __device__ __forceinline__ Vec3 make(T value, T value2, T value3);
    static __device__ __forceinline__ Vec4 make(T value, T value2, T value3, T value4);
};

template <>
struct VecMaker<f32>
{
    using Vec2 = typename VectorTrait<float, 2>::type;
    using Vec3 = typename VectorTrait<float, 3>::type;
    using Vec4 = typename VectorTrait<float, 4>::type;
    static __device__ __forceinline__ Vec2 make(f32 x, f32 y)
    {
        return make_float2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(f32 x, f32 y, f32 z)
    {
        return make_float3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(f32 x, f32 y, f32 z, f32 w)
    {
        return make_float4(x, y, z, w);
    }
};

template <>
struct VecMaker<f64>
{
    using Vec2 = typename VectorTrait<f64, 2>::type;
    using Vec3 = typename VectorTrait<f64, 3>::type;
    using Vec4 = typename VectorTrait<f64, 4>::type;
    static __device__ __forceinline__ Vec2 make(f64 x, f64 y)
    {
        return make_double2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(f32 x, f32 y, f32 z)
    {
        return make_double3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(f64 x, f64 y, f64 z, f64 w)
    {
        return make_double4(x, y, z, w);
    }
};

template <>
struct VecMaker<f16>
{
    using Vec2 = typename VectorTrait<f16, 2>::type;
    static __device__ __forceinline__ Vec2 make(f16 x, f16 y)
    {
        return make_half2(x, y);
    }
};

template <>
struct VecMaker<bf16>
{
    using Vec2 = typename VectorTrait<bf16, 2>::type;
    static __device__ __forceinline__ Vec2 make(bf16 x, bf16 y)
    {
        return make_bfloat162(x, y);
    }
};

template <>
struct VecMaker<i8>
{
    using Vec2 = typename VectorTrait<i8, 2>::type;
    using Vec3 = typename VectorTrait<i8, 3>::type;
    using Vec4 = typename VectorTrait<i8, 4>::type;
    static __device__ __forceinline__ Vec2 make(i8 x, i8 y)
    {
        return make_char2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(i8 x, i8 y, i8 z)
    {
        return make_char3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(i8 x, i8 y, i8 z, i8 w)
    {
        return make_char4(x, y, z, w);
    }
};

template <>
struct VecMaker<u8>
{
    using Vec2 = typename VectorTrait<u8, 2>::type;
    using Vec3 = typename VectorTrait<u8, 3>::type;
    using Vec4 = typename VectorTrait<u8, 4>::type;
    static __device__ __forceinline__ Vec2 make(u8 x, u8 y)
    {
        return make_uchar2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(u8 x, u8 y, u8 z)
    {
        return make_uchar3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(u8 x, u8 y, u8 z, u8 w)
    {
        return make_uchar4(x, y, z, w);
    }
};

template <>
struct VecMaker<i16>
{
    using Vec2 = typename VectorTrait<i16, 2>::type;
    using Vec3 = typename VectorTrait<i16, 3>::type;
    using Vec4 = typename VectorTrait<i16, 4>::type;
    static __device__ __forceinline__ Vec2 make(i16 x, i16 y)
    {
        return make_short2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(i16 x, i16 y, i16 z)
    {
        return make_short3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(i16 x, i16 y, i16 z, i16 w)
    {
        return make_short4(x, y, z, w);
    }
};

template <>
struct VecMaker<u16>
{
    using Vec2 = typename VectorTrait<u16, 2>::type;
    using Vec3 = typename VectorTrait<u16, 3>::type;
    using Vec4 = typename VectorTrait<u16, 4>::type;
    static __device__ __forceinline__ Vec2 make(u16 x, u16 y)
    {
        return make_ushort2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(u16 x, u16 y, u16 z)
    {
        return make_ushort3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(u16 x, u16 y, u16 z, u16 w)
    {
        return make_ushort4(x, y, z, w);
    }
};

template <>
struct VecMaker<i32>
{
    using Vec2 = typename VectorTrait<i32, 2>::type;
    using Vec3 = typename VectorTrait<i32, 3>::type;
    using Vec4 = typename VectorTrait<i32, 4>::type;
    static __device__ __forceinline__ Vec2 make(i32 x, i32 y)
    {
        return make_int2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(i32 x, i32 y, i32 z)
    {
        return make_int3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(i32 x, i32 y, i32 z, i32 w)
    {
        return make_int4(x, y, z, w);
    }
};

template <>
struct VecMaker<u32>
{
    using Vec2 = typename VectorTrait<u32, 2>::type;
    using Vec3 = typename VectorTrait<u32, 3>::type;
    using Vec4 = typename VectorTrait<u32, 4>::type;
    static __device__ __forceinline__ Vec2 make(u32 x, u32 y)
    {
        return make_uint2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(u32 x, u32 y, u32 z)
    {
        return make_uint3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(u32 x, u32 y, u32 z, u32 w)
    {
        return make_uint4(x, y, z, w);
    }
};

template <>
struct VecMaker<i64>
{
    using Vec2 = typename VectorTrait<i64, 2>::type;
    using Vec3 = typename VectorTrait<i64, 3>::type;
    using Vec4 = typename VectorTrait<i64, 4>::type;
    static __device__ __forceinline__ Vec2 make(i64 x, i64 y)
    {
        return make_long2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(i64 x, i64 y, i64 z)
    {
        return make_long3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(i64 x, i64 y, i64 z, i64 w)
    {
        return make_long4(x, y, z, w);
    }
};

template <>
struct VecMaker<u64>
{
    using Vec2 = typename VectorTrait<u64, 2>::type;
    using Vec3 = typename VectorTrait<u64, 3>::type;
    using Vec4 = typename VectorTrait<u64, 4>::type;
    static __device__ __forceinline__ Vec2 make(u64 x, u64 y)
    {
        return make_ulong2(x, y);
    }
    static __device__ __forceinline__ Vec3 make(u64 x, u64 y, u64 z)
    {
        return make_ulong3(x, y, z);
    }
    static __device__ __forceinline__ Vec4 make(u64 x, u64 y, u64 z, u64 w)
    {
        return make_ulong4(x, y, z, w);
    }
};
