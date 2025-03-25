#include "binary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"

template <typename LHS, typename RHS>
struct Mul
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a && b;
        }
        else
        {
            return cast<LHS, Output>(a) * cast<RHS, Output>(b);
        }
    }
};

DEFINE_BINARY_KERNEL(mul_bool_bool, bool, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_i8, bool, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_i16, bool, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_i32, bool, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_i64, bool, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_u8, bool, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_u16, bool, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_u32, bool, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_u64, bool, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_f16, bool, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_bf16, bool, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_f32, bool, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bool_f64, bool, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_i8_bool, i8, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_i8, i8, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_i16, i8, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_i32, i8, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_i64, i8, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_u8, i8, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_u16, i8, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_u32, i8, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_u64, i8, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_f16, i8, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_bf16, i8, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_f32, i8, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i8_f64, i8, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_i16_bool, i16, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_i8, i16, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_i16, i16, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_i32, i16, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_i64, i16, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_u8, i16, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_u16, i16, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_u32, i16, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_u64, i16, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_f16, i16, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_bf16, i16, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_f32, i16, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i16_f64, i16, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_i32_bool, i32, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_i8, i32, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_i16, i32, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_i32, i32, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_i64, i32, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_u8, i32, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_u16, i32, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_u32, i32, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_u64, i32, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_f16, i32, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_bf16, i32, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_f32, i32, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i32_f64, i32, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_i64_bool, i64, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_i8, i64, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_i16, i64, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_i32, i64, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_i64, i64, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_u8, i64, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_u16, i64, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_u32, i64, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_u64, i64, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_f16, i64, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_bf16, i64, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_f32, i64, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_i64_f64, i64, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_u8_bool, u8, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_i8, u8, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_i16, u8, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_i32, u8, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_i64, u8, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_u8, u8, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_u16, u8, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_u32, u8, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_u64, u8, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_f16, u8, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_bf16, u8, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_f32, u8, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u8_f64, u8, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_u16_bool, u16, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_i8, u16, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_i16, u16, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_i32, u16, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_i64, u16, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_u8, u16, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_u16, u16, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_u32, u16, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_u64, u16, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_f16, u16, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_bf16, u16, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_f32, u16, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u16_f64, u16, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_u32_bool, u32, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_i8, u32, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_i16, u32, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_i32, u32, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_i64, u32, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_u8, u32, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_u16, u32, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_u32, u32, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_u64, u32, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_f16, u32, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_bf16, u32, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_f32, u32, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u32_f64, u32, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_u64_bool, u64, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_i8, u64, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_i16, u64, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_i32, u64, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_i64, u64, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_u8, u64, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_u16, u64, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_u32, u64, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_u64, u64, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_f16, u64, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_bf16, u64, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_f32, u64, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_u64_f64, u64, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_f16_bool, f16, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_i8, f16, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_i16, f16, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_i32, f16, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_i64, f16, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_u8, f16, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_u16, f16, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_u32, f16, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_u64, f16, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_f16, f16, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_bf16, f16, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_f32, f16, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f16_f64, f16, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_bf16_bool, bf16, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_i8, bf16, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_i16, bf16, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_i32, bf16, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_i64, bf16, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_u8, bf16, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_u16, bf16, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_u32, bf16, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_u64, bf16, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_f16, bf16, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_f32, bf16, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_bf16_f64, bf16, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_f32_bool, f32, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_i8, f32, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_i16, f32, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_i32, f32, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_i64, f32, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_u8, f32, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_u16, f32, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_u32, f32, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_u64, f32, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_f16, f32, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_bf16, f32, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_f32, f32, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f32_f64, f32, f64, Mul, NormalOutPromote);

DEFINE_BINARY_KERNEL(mul_f64_bool, f64, bool, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_i8, f64, i8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_i16, f64, i16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_i32, f64, i32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_i64, f64, i64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_u8, f64, u8, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_u16, f64, u16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_u32, f64, u32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_u64, f64, u64, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_f16, f64, f16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_bf16, f64, bf16, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_f32, f64, f32, Mul, NormalOutPromote);
DEFINE_BINARY_KERNEL(mul_f64_f64, f64, f64, Mul, NormalOutPromote);
