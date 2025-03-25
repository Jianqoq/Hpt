#pragma once
#include "type_alias.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T, int N>
struct VectorTrait;

template <>
struct VectorTrait<f32, 2>
{
    using type = float2;
};

template <>
struct VectorTrait<f32, 3>
{
    using type = float3;
};

template <>
struct VectorTrait<f32, 4>
{
    using type = float4;
};

template <>
struct VectorTrait<f64, 2>
{
    using type = double2;
};

template <>
struct VectorTrait<f64, 3>
{
    using type = double3;
};

template <>
struct VectorTrait<f64, 4>
{
    using type = double4;
};

template <>
struct VectorTrait<bf16, 2>
{
    using type = __nv_bfloat162;
};

template <>
struct VectorTrait<f16, 2>
{
    using type = half2;
};

template <>
struct VectorTrait<i8, 2>
{
    using type = char2;
};

template <>
struct VectorTrait<i8, 3>
{
    using type = char3;
};

template <>
struct VectorTrait<i8, 4>
{
    using type = char4;
};

template <>
struct VectorTrait<u8, 2>
{
    using type = uchar2;
};

template <>
struct VectorTrait<u8, 3>
{
    using type = uchar3;
};

template <>
struct VectorTrait<u8, 4>
{
    using type = uchar4;
};

template <>
struct VectorTrait<i16, 2>
{
    using type = short2;
};

template <>
struct VectorTrait<i16, 3>
{
    using type = short3;
};

template <>
struct VectorTrait<i16, 4>
{
    using type = short4;
};

template <>
struct VectorTrait<u16, 2>
{
    using type = ushort2;
};

template <>
struct VectorTrait<u16, 3>
{
    using type = ushort3;
};

template <>
struct VectorTrait<u16, 4>
{
    using type = ushort4;
};

template <>
struct VectorTrait<i32, 2>
{
    using type = int2;
};

template <>
struct VectorTrait<i32, 3>
{
    using type = int3;
};

template <>
struct VectorTrait<i32, 4>
{
    using type = int4;
};

template <>
struct VectorTrait<u32, 2>
{
    using type = uint2;
};

template <>
struct VectorTrait<u32, 3>
{
    using type = uint3;
};

template <>
struct VectorTrait<u32, 4>
{
    using type = uint4;
};

template <>
struct VectorTrait<i64, 2>
{
    using type = long2;
};

template <>
struct VectorTrait<i64, 3>
{
    using type = long3;
};

template <>
struct VectorTrait<i64, 4>
{
    using type = long4;
};

template <>
struct VectorTrait<u64, 2>
{
    using type = ulong2;
};

template <>
struct VectorTrait<u64, 3>
{
    using type = ulong3;
};

template <>
struct VectorTrait<u64, 4>
{
    using type = ulong4;
};



