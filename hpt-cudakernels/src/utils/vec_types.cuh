#pragma once
#include "type_alias.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "extra_vecs.cuh"

template <typename T, int N>
struct VectorTrait;

template <>
struct VectorTrait<bool, 1>
{
    using type = bool;
};

template <>
struct VectorTrait<bool, 2>
{
    using type = bool2;
};

template <>
struct VectorTrait<bool, 3>
{
    using type = bool3;
};

template <>
struct VectorTrait<bool, 4>
{
    using type = bool4;
};

template <>
struct VectorTrait<f32, 1>
{
    using type = float;
};

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
struct VectorTrait<f64, 1>
{
    using type = double;
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
struct VectorTrait<bf16, 1>
{
    using type = __nv_bfloat16;
};

template <>
struct VectorTrait<bf16, 2>
{
    using type = __nv_bfloat162;
};

template <>
struct VectorTrait<bf16, 3>
{
    using type = bf163;
};

template <>
struct VectorTrait<bf16, 4>
{
    using type = bf164;
};

template <>
struct VectorTrait<f16, 1>
{
    using type = half;
};

template <>
struct VectorTrait<f16, 2>
{
    using type = half2;
};

template <>
struct VectorTrait<f16, 3>
{
    using type = half3;
};

template <>
struct VectorTrait<f16, 4>
{
    using type = half4;
};

template <>
struct VectorTrait<i8, 1>
{
    using type = i8;
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
struct VectorTrait<u8, 1>
{
    using type = u8;
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
struct VectorTrait<i16, 1>
{
    using type = short;
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
struct VectorTrait<u16, 1>
{
    using type = u16;
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
struct VectorTrait<i32, 1>
{
    using type = int;
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
struct VectorTrait<u32, 1>
{
    using type = u32;
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
struct VectorTrait<i64, 1>
{
    using type = i64;
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
struct VectorTrait<u64, 1>
{
    using type = u64;
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
