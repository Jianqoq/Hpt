#pragma once
#include "type_alias.cuh"
#include "vec_types.cuh"

template <typename T>
struct VecMaker;

template <>
struct VecMaker<bool>
{
    using Vec1 = typename VectorTrait<bool, 1>::type;
    using Vec2 = typename VectorTrait<bool, 2>::type;
    using Vec3 = typename VectorTrait<bool, 3>::type;
    using Vec4 = typename VectorTrait<bool, 4>::type;
    static __device__ __forceinline__ Vec1 make(bool x)
    {
        return x;
    }
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<bool, N>::type make_unaligned(bool *ptr);

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<bool, N>::type make(bool *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(bool *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(bool *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(bool *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(bool *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }
    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(bool *ptr)
    {
        return ptr[0];
    }
    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(bool *ptr)
    {
        Vec2 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        return vec;
    }
    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(bool *ptr)
    {
        Vec3 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        return vec;
    }
    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(bool *ptr)
    {
        Vec4 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        vec.w = ptr[3];
        return vec;
    }
};

template <>
struct VecMaker<f32>
{
    using Vec1 = typename VectorTrait<float, 1>::type;
    using Vec2 = typename VectorTrait<float, 2>::type;
    using Vec3 = typename VectorTrait<float, 3>::type;
    using Vec4 = typename VectorTrait<float, 4>::type;
    static __device__ __forceinline__ Vec1 make(f32 x)
    {
        return x;
    }
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

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f32, N>::type make(f32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(f32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(f32 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(f32 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(f32 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f32, N>::type make_unaligned(f32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(f32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(f32 *ptr)
    {
        return make_float2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(f32 *ptr)
    {
        return make_float3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(f32 *ptr)
    {
        return make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<f64>
{
    using Vec1 = typename VectorTrait<f64, 1>::type;
    using Vec2 = typename VectorTrait<f64, 2>::type;
    using Vec3 = typename VectorTrait<f64, 3>::type;
    using Vec4 = typename VectorTrait<f64, 4>::type;
    static __device__ __forceinline__ Vec1 make(f64 x)
    {
        return x;
    }
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

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f64, N>::type make(f64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(f64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(f64 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(f64 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(f64 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f64, N>::type make_unaligned(f64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(f64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(f64 *ptr)
    {
        return make_double2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(f64 *ptr)
    {
        return make_double3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(f64 *ptr)
    {
        return make_double4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<f16>
{
    using Vec1 = typename VectorTrait<f16, 1>::type;
    using Vec2 = typename VectorTrait<f16, 2>::type;
    using Vec3 = typename VectorTrait<f16, 3>::type;
    using Vec4 = typename VectorTrait<f16, 4>::type;
    static __device__ __forceinline__ Vec1 make(f16 x)
    {
        return x;
    }
    static __device__ __forceinline__ Vec2 make(f16 x, f16 y)
    {
        return make_half2(x, y);
    }
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f16, N>::type make_unaligned(f16 *ptr);

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<f16, N>::type make(f16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(f16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(f16 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(f16 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(f16 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(f16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(f16 *ptr)
    {
        return make_half2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(f16 *ptr)
    {
        Vec3 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        return vec;
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(f16 *ptr)
    {
        Vec4 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        vec.w = ptr[3];
        return vec;
    }
};

template <>
struct VecMaker<bf16>
{
    using Vec1 = typename VectorTrait<bf16, 1>::type;
    using Vec2 = typename VectorTrait<bf16, 2>::type;
    using Vec3 = typename VectorTrait<bf16, 3>::type;
    using Vec4 = typename VectorTrait<bf16, 4>::type;
    static __device__ __forceinline__ Vec1 make(bf16 x)
    {
        return x;
    }
    static __device__ __forceinline__ Vec2 make(bf16 x, bf16 y)
    {
        return make_bfloat162(x, y);
    }
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<bf16, N>::type make(bf16 *ptr);

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<bf16, N>::type make_unaligned(bf16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(bf16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(bf16 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(bf16 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(bf16 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(bf16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(bf16 *ptr)
    {
        return make_bfloat162(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(bf16 *ptr)
    {
        Vec3 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        return vec;
    }
    
    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(bf16 *ptr)
    {
        Vec4 vec;
        vec.x = ptr[0];
        vec.y = ptr[1];
        vec.z = ptr[2];
        vec.w = ptr[3];
        return vec;
    }
};

template <>
struct VecMaker<i8>
{
    using Vec1 = typename VectorTrait<i8, 1>::type;
    using Vec2 = typename VectorTrait<i8, 2>::type;
    using Vec3 = typename VectorTrait<i8, 3>::type;
    using Vec4 = typename VectorTrait<i8, 4>::type;
    static __device__ __forceinline__ Vec1 make(i8 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i8, N>::type make(i8 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(i8 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(i8 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(i8 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(i8 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i8, N>::type make_unaligned(i8 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(i8 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(i8 *ptr)
    {
        return make_char2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(i8 *ptr)
    {
        return make_char3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(i8 *ptr)
    {
        return make_char4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<u8>
{
    using Vec1 = typename VectorTrait<u8, 1>::type;
    using Vec2 = typename VectorTrait<u8, 2>::type;
    using Vec3 = typename VectorTrait<u8, 3>::type;
    using Vec4 = typename VectorTrait<u8, 4>::type;
    static __device__ __forceinline__ Vec1 make(u8 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u8, N>::type make(u8 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(u8 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(u8 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(u8 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(u8 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u8, N>::type make_unaligned(u8 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(u8 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(u8 *ptr)
    {
        return make_uchar2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(u8 *ptr)
    {
        return make_uchar3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(u8 *ptr)
    {
        return make_uchar4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<i16>
{
    using Vec1 = typename VectorTrait<i16, 1>::type;
    using Vec2 = typename VectorTrait<i16, 2>::type;
    using Vec3 = typename VectorTrait<i16, 3>::type;
    using Vec4 = typename VectorTrait<i16, 4>::type;
    static __device__ __forceinline__ Vec1 make(i16 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i16, N>::type make(i16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(i16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(i16 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(i16 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(i16 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i16, N>::type make_unaligned(i16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(i16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(i16 *ptr)
    {
        return make_short2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(i16 *ptr)
    {
        return make_short3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(i16 *ptr)
    {
        return make_short4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<u16>
{
    using Vec1 = typename VectorTrait<u16, 1>::type;
    using Vec2 = typename VectorTrait<u16, 2>::type;
    using Vec3 = typename VectorTrait<u16, 3>::type;
    using Vec4 = typename VectorTrait<u16, 4>::type;
    static __device__ __forceinline__ Vec1 make(u16 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u16, N>::type make(u16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(u16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(u16 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(u16 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(u16 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u16, N>::type make_unaligned(u16 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(u16 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(u16 *ptr)
    {
        return make_ushort2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(u16 *ptr)
    {
        return make_ushort3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(u16 *ptr)
    {
        return make_ushort4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<i32>
{
    using Vec1 = typename VectorTrait<i32, 1>::type;
    using Vec2 = typename VectorTrait<i32, 2>::type;
    using Vec3 = typename VectorTrait<i32, 3>::type;
    using Vec4 = typename VectorTrait<i32, 4>::type;
    static __device__ __forceinline__ Vec1 make(i32 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i32, N>::type make(i32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(i32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(i32 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(i32 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(i32 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i32, N>::type make_unaligned(i32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(i32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(i32 *ptr)
    {
        return make_int2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(i32 *ptr)
    {
        return make_int3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(i32 *ptr)
    {
        return make_int4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<u32>
{
    using Vec1 = typename VectorTrait<u32, 1>::type;
    using Vec2 = typename VectorTrait<u32, 2>::type;
    using Vec3 = typename VectorTrait<u32, 3>::type;
    using Vec4 = typename VectorTrait<u32, 4>::type;
    static __device__ __forceinline__ Vec1 make(u32 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u32, N>::type make(u32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(u32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(u32 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(u32 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(u32 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u32, N>::type make_unaligned(u32 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(u32 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(u32 *ptr)
    {
        return make_uint2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(u32 *ptr)
    {
        return make_uint3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(u32 *ptr)
    {
        return make_uint4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<i64>
{
    using Vec1 = typename VectorTrait<i64, 1>::type;
    using Vec2 = typename VectorTrait<i64, 2>::type;
    using Vec3 = typename VectorTrait<i64, 3>::type;
    using Vec4 = typename VectorTrait<i64, 4>::type;
    static __device__ __forceinline__ Vec1 make(i64 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i64, N>::type make(i64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(i64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(i64 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(i64 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(i64 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<i64, N>::type make_unaligned(i64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(i64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(i64 *ptr)
    {
        return make_long2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(i64 *ptr)
    {
        return make_long3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(i64 *ptr)
    {
        return make_long4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};

template <>
struct VecMaker<u64>
{
    using Vec1 = typename VectorTrait<u64, 1>::type;
    using Vec2 = typename VectorTrait<u64, 2>::type;
    using Vec3 = typename VectorTrait<u64, 3>::type;
    using Vec4 = typename VectorTrait<u64, 4>::type;
    static __device__ __forceinline__ Vec1 make(u64 x)
    {
        return x;
    }
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
    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u64, N>::type make(u64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make<1>(u64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make<2>(u64 *ptr)
    {
        return *reinterpret_cast<Vec2 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec3 make<3>(u64 *ptr)
    {
        return *reinterpret_cast<Vec3 *>(ptr);
    }

    template <>
    static __device__ __forceinline__ Vec4 make<4>(u64 *ptr)
    {
        return *reinterpret_cast<Vec4 *>(ptr);
    }

    template <int N>
    static __device__ __forceinline__ typename VectorTrait<u64, N>::type make_unaligned(u64 *ptr);

    template <>
    static __device__ __forceinline__ Vec1 make_unaligned<1>(u64 *ptr)
    {
        return ptr[0];
    }

    template <>
    static __device__ __forceinline__ Vec2 make_unaligned<2>(u64 *ptr)
    {
        return make_ulong2(ptr[0], ptr[1]);
    }

    template <>
    static __device__ __forceinline__ Vec3 make_unaligned<3>(u64 *ptr)
    {
        return make_ulong3(ptr[0], ptr[1], ptr[2]);
    }

    template <>
    static __device__ __forceinline__ Vec4 make_unaligned<4>(u64 *ptr)
    {
        return make_ulong4(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
};
