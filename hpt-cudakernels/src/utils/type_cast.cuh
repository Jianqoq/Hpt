
#pragma once

#include <stdint.h>

#define case(IN_TYPE, OUT_TYPE, RUST_IN_TYPE, RUST_OUT_TYPE)                 \
    if constexpr (std::is_same_v<T, IN_TYPE> && std::is_same_v<R, OUT_TYPE>) \
    {                                                                        \
        return RUST_IN_TYPE##_to_##RUST_OUT_TYPE(x);                         \
    }

#define bool_to_bool(x) (x)
#define bool_to_i8(x) ((x) ? 1 : 0)
#define bool_to_i16(x) ((x) ? 1 : 0)
#define bool_to_i32(x) ((x) ? 1 : 0)
#define bool_to_i64(x) ((x) ? 1 : 0)
#define bool_to_u8(x) ((x) ? 1 : 0)
#define bool_to_u16(x) ((x) ? 1 : 0)
#define bool_to_u32(x) ((x) ? 1 : 0)
#define bool_to_u64(x) ((x) ? 1 : 0)
#define bool_to_f32(x) ((x) ? 1.0f : 0.0f)
#define bool_to_f64(x) ((x) ? 1.0 : 0.0)
#define bool_to_f16(x) __float2half_rn((x) ? 1.0f : 0.0f)
#define bool_to_bf16(x) __float2bfloat16_rn((x) ? 1.0f : 0.0f)

#define i8_to_bool(x) ((x) != 0)
#define i8_to_i8(x) (x)
#define i8_to_i16(x) ((int16_t)(x))
#define i8_to_i32(x) ((int32_t)(x))
#define i8_to_i64(x) ((int64_t)(x))
#define i8_to_u8(x) ((uint8_t)(x))
#define i8_to_u16(x) ((uint16_t)(x))
#define i8_to_u32(x) ((uint32_t)(x))
#define i8_to_u64(x) ((uint64_t)(x))
#define i8_to_f32(x) ((float)(x))
#define i8_to_f64(x) ((double)(x))
#define i8_to_f16(x) __float2half_rn((float)(x))
#define i8_to_bf16(x) __float2bfloat16_rn((float)(x))

#define i16_to_bool(x) ((x) != 0)
#define i16_to_i8(x) ((int8_t)(x))
#define i16_to_i16(x) (x)
#define i16_to_i32(x) ((int32_t)(x))
#define i16_to_i64(x) ((int64_t)(x))
#define i16_to_u8(x) ((uint8_t)(x))
#define i16_to_u16(x) ((uint16_t)(x))
#define i16_to_u32(x) ((uint32_t)(x))
#define i16_to_u64(x) ((uint64_t)(x))
#define i16_to_f32(x) ((float)(x))
#define i16_to_f64(x) ((double)(x))
#define i16_to_f16(x) __float2half_rn((float)(x))
#define i16_to_bf16(x) __float2bfloat16_rn((float)(x))

#define i32_to_bool(x) ((x) != 0)
#define i32_to_i8(x) ((int8_t)(x))
#define i32_to_i16(x) ((int16_t)(x))
#define i32_to_i32(x) (x)
#define i32_to_i64(x) ((int64_t)(x))
#define i32_to_u8(x) ((uint8_t)(x))
#define i32_to_u16(x) ((uint16_t)(x))
#define i32_to_u32(x) ((uint32_t)(x))
#define i32_to_u64(x) ((uint64_t)(x))
#define i32_to_f32(x) ((float)(x))
#define i32_to_f64(x) ((double)(x))
#define i32_to_f16(x) __float2half_rn((float)(x))
#define i32_to_bf16(x) __float2bfloat16_rn((float)(x))

#define i64_to_bool(x) ((x) != 0)
#define i64_to_i8(x) ((int8_t)(x))
#define i64_to_i16(x) ((int16_t)(x))
#define i64_to_i32(x) ((int32_t)(x))
#define i64_to_i64(x) (x)
#define i64_to_u8(x) ((uint8_t)(x))
#define i64_to_u16(x) ((uint16_t)(x))
#define i64_to_u32(x) ((uint32_t)(x))
#define i64_to_u64(x) ((uint64_t)(x))
#define i64_to_f32(x) ((float)(x))
#define i64_to_f64(x) ((double)(x))
#define i64_to_f16(x) __double2half((double)(x))
#define i64_to_bf16(x) __double2bfloat16((double)(x))

#define u8_to_bool(x) ((x) != 0)
#define u8_to_i8(x) ((int8_t)(x))
#define u8_to_i16(x) ((int16_t)(x))
#define u8_to_i32(x) ((int32_t)(x))
#define u8_to_i64(x) ((int64_t)(x))
#define u8_to_u8(x) (x)
#define u8_to_u16(x) ((uint16_t)(x))
#define u8_to_u32(x) ((uint32_t)(x))
#define u8_to_u64(x) ((uint64_t)(x))
#define u8_to_f32(x) ((float)(x))
#define u8_to_f64(x) ((double)(x))
#define u8_to_f16(x) __float2half_rn((float)(x))
#define u8_to_bf16(x) __float2bfloat16_rn((float)(x))

#define u16_to_bool(x) ((x) != 0)
#define u16_to_i8(x) ((int8_t)(x))
#define u16_to_i16(x) ((int16_t)(x))
#define u16_to_i32(x) ((int32_t)(x))
#define u16_to_i64(x) ((int64_t)(x))
#define u16_to_u8(x) ((uint8_t)(x))
#define u16_to_u16(x) (x)
#define u16_to_u32(x) ((uint32_t)(x))
#define u16_to_u64(x) ((uint64_t)(x))
#define u16_to_f32(x) ((float)(x))
#define u16_to_f64(x) ((double)(x))
#define u16_to_f16(x) __float2half_rn((float)(x))
#define u16_to_bf16(x) __float2bfloat16_rn((float)(x))

#define u32_to_bool(x) ((x) != 0)
#define u32_to_i8(x) ((int8_t)(x))
#define u32_to_i16(x) ((int16_t)(x))
#define u32_to_i32(x) ((int32_t)(x))
#define u32_to_i64(x) ((int64_t)(x))
#define u32_to_u8(x) ((uint8_t)(x))
#define u32_to_u16(x) ((uint16_t)(x))
#define u32_to_u32(x) (x)
#define u32_to_u64(x) ((uint64_t)(x))
#define u32_to_f32(x) ((float)(x))
#define u32_to_f64(x) ((double)(x))
#define u32_to_f16(x) __float2half_rn((float)(x))
#define u32_to_bf16(x) __float2bfloat16_rn((float)(x))

#define u64_to_bool(x) ((x) != 0)
#define u64_to_i8(x) ((int8_t)(x))
#define u64_to_i16(x) ((int16_t)(x))
#define u64_to_i32(x) ((int32_t)(x))
#define u64_to_i64(x) ((int64_t)(x))
#define u64_to_u8(x) ((uint8_t)(x))
#define u64_to_u16(x) ((uint16_t)(x))
#define u64_to_u32(x) ((uint32_t)(x))
#define u64_to_u64(x) (x)
#define u64_to_f32(x) ((float)(x))
#define u64_to_f64(x) ((double)(x))
#define u64_to_f16(x) __double2half((double)(x))
#define u64_to_bf16(x) __double2bfloat16((double)(x))

#define f32_to_bool(x) ((x) != 0.0f)
#define f32_to_i8(x) ((int8_t)(x))
#define f32_to_i16(x) ((int16_t)(x))
#define f32_to_i32(x) ((int32_t)(x))
#define f32_to_i64(x) ((int64_t)(x))
#define f32_to_u8(x) ((uint8_t)(x))
#define f32_to_u16(x) ((uint16_t)(x))
#define f32_to_u32(x) ((uint32_t)(x))
#define f32_to_u64(x) ((uint64_t)(x))
#define f32_to_f32(x) (x)
#define f32_to_f64(x) ((double)(x))
#define f32_to_f16(x) __float2half_rn((x))
#define f32_to_bf16(x) __float2bfloat16_rn((x))

#define f64_to_bool(x) ((x) != 0.0)
#define f64_to_i8(x) ((int8_t)(x))
#define f64_to_i16(x) ((int16_t)(x))
#define f64_to_i32(x) ((int32_t)(x))
#define f64_to_i64(x) ((int64_t)(x))
#define f64_to_u8(x) ((uint8_t)(x))
#define f64_to_u16(x) ((uint16_t)(x))
#define f64_to_u32(x) ((uint32_t)(x))
#define f64_to_u64(x) ((uint64_t)(x))
#define f64_to_f32(x) ((float)(x))
#define f64_to_f64(x) (x)
#define f64_to_f16(x) __double2half((x))
#define f64_to_bf16(x) __double2bfloat16((x))

#define f16_to_bool(x) (__half2char_rz(x) != 0)
#define f16_to_i8(x) (__half2char_rz(x))
#define f16_to_i16(x) (__half2short_rn(x))
#define f16_to_i32(x) (__half2int_rn(x))
#define f16_to_i64(x) (__half2ll_rn(x))
#define f16_to_u8(x) (__half2uchar_rz(x))
#define f16_to_u16(x) (__half2ushort_rn(x))
#define f16_to_u32(x) (__half2uint_rn(x))
#define f16_to_u64(x) (__half2ull_rn(x))
#define f16_to_f32(x) (__half2float(x))
#define f16_to_f64(x) ((double)__half2float(x))
#define f16_to_bf16(x) (__float2bfloat16_rn(__half2float(x)))
#define f16_to_f16(x) (x)

#define bf16_to_bool(x) (__bfloat162char_rz(x) != 0)
#define bf16_to_i8(x) (__bfloat162char_rz(x))
#define bf16_to_i16(x) (__bfloat162short_rn(x))
#define bf16_to_i32(x) (__bfloat162int_rn(x))
#define bf16_to_i64(x) (__bfloat162ll_rn(x))
#define bf16_to_u8(x) (__bfloat162uchar_rz(x))
#define bf16_to_u16(x) (__bfloat162ushort_rn(x))
#define bf16_to_u32(x) (__bfloat162uint_rn(x))
#define bf16_to_u64(x) (__bfloat162ull_rn(x))
#define bf16_to_f32(x) (__bfloat162float(x))
#define bf16_to_f64(x) ((double)__bfloat162float(x))
#define bf16_to_bf16(x) (x)
#define bf16_to_f16(x) (__float2half_rn(__bfloat162float(x)))

template <typename T, typename R>
__device__ __forceinline__ R cast(T x)
{
    case(bool, bool, bool, bool);
    case(bool, int8_t, bool, i8);
    case(bool, int16_t, bool, i16);
    case(bool, int32_t, bool, i32);
    case(bool, int64_t, bool, i64);
    case(bool, uint8_t, bool, u8);
    case(bool, uint16_t, bool, u16);
    case(bool, uint32_t, bool, u32);
    case(bool, uint64_t, bool, u64);
    case(bool, float, bool, f32);
    case(bool, double, bool, f64);
    case(bool, __half, bool, f16);
    case(bool, __nv_bfloat16, bool, bf16);

    case(int8_t, bool, i8, bool);
    case(int8_t, int8_t, i8, i8);
    case(int8_t, int16_t, i8, i16);
    case(int8_t, int32_t, i8, i32);
    case(int8_t, int64_t, i8, i64);
    case(int8_t, uint8_t, i8, u8);
    case(int8_t, uint16_t, i8, u16);
    case(int8_t, uint32_t, i8, u32);
    case(int8_t, uint64_t, i8, u64);
    case(int8_t, float, i8, f32);
    case(int8_t, double, i8, f64);
    case(int8_t, __half, i8, f16);
    case(int8_t, __nv_bfloat16, i8, bf16);

    case(int16_t, bool, i16, bool);
    case(int16_t, int8_t, i16, i8);
    case(int16_t, int16_t, i16, i16);
    case(int16_t, int32_t, i16, i32);
    case(int16_t, int64_t, i16, i64);
    case(int16_t, uint8_t, i16, u8);
    case(int16_t, uint16_t, i16, u16);
    case(int16_t, uint32_t, i16, u32);
    case(int16_t, uint64_t, i16, u64);
    case(int16_t, float, i16, f32);
    case(int16_t, double, i16, f64);
    case(int16_t, __half, i16, f16);
    case(int16_t, __nv_bfloat16, i16, bf16);

    case(int32_t, bool, i32, bool);
    case(int32_t, int8_t, i32, i8);
    case(int32_t, int16_t, i32, i16);
    case(int32_t, int32_t, i32, i32);
    case(int32_t, int64_t, i32, i64);
    case(int32_t, uint8_t, i32, u8);
    case(int32_t, uint16_t, i32, u16);
    case(int32_t, uint32_t, i32, u32);
    case(int32_t, uint64_t, i32, u64);
    case(int32_t, float, i32, f32);
    case(int32_t, double, i32, f64);
    case(int32_t, __half, i32, f16);
    case(int32_t, __nv_bfloat16, i32, bf16);

    case(int64_t, bool, i64, bool);
    case(int64_t, int8_t, i64, i8);
    case(int64_t, int16_t, i64, i16);
    case(int64_t, int32_t, i64, i32);
    case(int64_t, int64_t, i64, i64);
    case(int64_t, uint8_t, i64, u8);
    case(int64_t, uint16_t, i64, u16);
    case(int64_t, uint32_t, i64, u32);
    case(int64_t, uint64_t, i64, u64);
    case(int64_t, float, i64, f32);
    case(int64_t, double, i64, f64);
    case(int64_t, __half, i64, f16);
    case(int64_t, __nv_bfloat16, i64, bf16);

    case(uint8_t, bool, u8, bool);
    case(uint8_t, int8_t, u8, i8);
    case(uint8_t, int16_t, u8, i16);
    case(uint8_t, int32_t, u8, i32);
    case(uint8_t, int64_t, u8, i64);
    case(uint8_t, uint8_t, u8, u8);
    case(uint8_t, uint16_t, u8, u16);
    case(uint8_t, uint32_t, u8, u32);
    case(uint8_t, uint64_t, u8, u64);
    case(uint8_t, float, u8, f32);
    case(uint8_t, double, u8, f64);
    case(uint8_t, __half, u8, f16);
    case(uint8_t, __nv_bfloat16, u8, bf16);

    case(uint16_t, bool, u16, bool);
    case(uint16_t, int8_t, u16, i8);
    case(uint16_t, int16_t, u16, i16);
    case(uint16_t, int32_t, u16, i32);
    case(uint16_t, int64_t, u16, i64);
    case(uint16_t, uint8_t, u16, u8);
    case(uint16_t, uint16_t, u16, u16);
    case(uint16_t, uint32_t, u16, u32);
    case(uint16_t, uint64_t, u16, u64);
    case(uint16_t, float, u16, f32);
    case(uint16_t, double, u16, f64);
    case(uint16_t, __half, u16, f16);
    case(uint16_t, __nv_bfloat16, u16, bf16);

    case(uint32_t, bool, u32, bool);
    case(uint32_t, int8_t, u32, i8);
    case(uint32_t, int16_t, u32, i16);
    case(uint32_t, int32_t, u32, i32);
    case(uint32_t, int64_t, u32, i64);
    case(uint32_t, uint8_t, u32, u8);
    case(uint32_t, uint16_t, u32, u16);
    case(uint32_t, uint32_t, u32, u32);
    case(uint32_t, uint64_t, u32, u64);
    case(uint32_t, float, u32, f32);
    case(uint32_t, double, u32, f64);
    case(uint32_t, __half, u32, f16);
    case(uint32_t, __nv_bfloat16, u32, bf16);
    
    case(uint64_t, bool, u64, bool);
    case(uint64_t, int8_t, u64, i8);
    case(uint64_t, int16_t, u64, i16);
    case(uint64_t, int32_t, u64, i32);
    case(uint64_t, int64_t, u64, i64);
    case(uint64_t, uint8_t, u64, u8);
    case(uint64_t, uint16_t, u64, u16);
    case(uint64_t, uint32_t, u64, u32);
    case(uint64_t, uint64_t, u64, u64);
    case(uint64_t, float, u64, f32);
    case(uint64_t, double, u64, f64);
    case(uint64_t, __half, u64, f16);
    case(uint64_t, __nv_bfloat16, u64, bf16);
    
    case(float, bool, f32, bool);
    case(float, int8_t, f32, i8);
    case(float, int16_t, f32, i16);
    case(float, int32_t, f32, i32);
    case(float, int64_t, f32, i64);
    case(float, uint8_t, f32, u8);
    case(float, uint16_t, f32, u16);
    case(float, uint32_t, f32, u32);
    case(float, uint64_t, f32, u64);
    case(float, float, f32, f32);
    case(float, double, f32, f64);
    case(float, __half, f32, f16);
    case(float, __nv_bfloat16, f32, bf16);

    case(double, bool, f64, bool);
    case(double, int8_t, f64, i8);
    case(double, int16_t, f64, i16);
    case(double, int32_t, f64, i32);
    case(double, int64_t, f64, i64);
    case(double, uint8_t, f64, u8);
    case(double, uint16_t, f64, u16);
    case(double, uint32_t, f64, u32);
    case(double, uint64_t, f64, u64);
    case(double, float, f64, f32);
    case(double, double, f64, f64);
    case(double, __half, f64, f16);
    case(double, __nv_bfloat16, f64, bf16);
    
    case(__half, bool, f16, bool);
    case(__half, int8_t, f16, i8);
    case(__half, int16_t, f16, i16);
    case(__half, int32_t, f16, i32);
    case(__half, int64_t, f16, i64);
    case(__half, uint8_t, f16, u8);
    case(__half, uint16_t, f16, u16);
    case(__half, uint32_t, f16, u32);
    case(__half, uint64_t, f16, u64);
    case(__half, float, f16, f32);
    case(__half, double, f16, f64);
    case(__half, __half, f16, f16);
    case(__half, __nv_bfloat16, f16, bf16);

    case(__nv_bfloat16, bool, bf16, bool);
    case(__nv_bfloat16, int8_t, bf16, i8);
    case(__nv_bfloat16, int16_t, bf16, i16);
    case(__nv_bfloat16, int32_t, bf16, i32);
    case(__nv_bfloat16, int64_t, bf16, i64);
    case(__nv_bfloat16, uint8_t, bf16, u8);
    case(__nv_bfloat16, uint16_t, bf16, u16);
    case(__nv_bfloat16, uint32_t, bf16, u32);
    case(__nv_bfloat16, uint64_t, bf16, u64);
    case(__nv_bfloat16, float, bf16, f32);
    case(__nv_bfloat16, double, bf16, f64);
    case(__nv_bfloat16, __half, bf16, f16);
    case(__nv_bfloat16, __nv_bfloat16, bf16, bf16);

    return R();
}