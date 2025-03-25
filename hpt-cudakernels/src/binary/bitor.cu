#include "binary_template.cuh"
#include "../utils/type_alias.cuh"
#include "../utils/promotion/promotes.cuh"

template <typename LHS, typename RHS>
struct Bitor
{
    using Output = typename NormalOutPromote<LHS, RHS>::Output;
    __device__ __forceinline__ Output operator()(LHS a, RHS b) const
    {
        if constexpr (std::is_same_v<Output, bool>)
        {
            return a | b;
        }
        else
        {
            return cast<LHS, Output>(a) | cast<RHS, Output>(b);
        }
    }
};

DEFINE_BINARY_KERNEL(bitor_bool_bool, bool, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_i8, bool, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_i16, bool, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_i32, bool, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_i64, bool, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_u8, bool, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_u16, bool, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_u32, bool, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_bool_u64, bool, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_i8_bool, i8, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_i8, i8, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_i16, i8, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_i32, i8, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_i64, i8, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_u8, i8, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_u16, i8, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_u32, i8, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i8_u64, i8, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_i16_bool, i16, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_i8, i16, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_i16, i16, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_i32, i16, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_i64, i16, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_u8, i16, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_u16, i16, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_u32, i16, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i16_u64, i16, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_i32_bool, i32, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_i8, i32, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_i16, i32, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_i32, i32, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_i64, i32, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_u8, i32, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_u16, i32, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_u32, i32, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i32_u64, i32, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_i64_bool, i64, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_i8, i64, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_i16, i64, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_i32, i64, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_i64, i64, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_u8, i64, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_u16, i64, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_u32, i64, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_i64_u64, i64, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_u8_bool, u8, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_i8, u8, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_i16, u8, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_i32, u8, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_i64, u8, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_u8, u8, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_u16, u8, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_u32, u8, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u8_u64, u8, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_u16_bool, u16, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_i8, u16, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_i16, u16, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_i32, u16, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_i64, u16, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_u8, u16, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_u16, u16, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_u32, u16, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u16_u64, u16, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_u32_bool, u32, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_i8, u32, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_i16, u32, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_i32, u32, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_i64, u32, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_u8, u32, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_u16, u32, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_u32, u32, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u32_u64, u32, u64, Bitor, NormalOutPromote);

DEFINE_BINARY_KERNEL(bitor_u64_bool, u64, bool, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_i8, u64, i8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_i16, u64, i16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_i32, u64, i32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_i64, u64, i64, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_u8, u64, u8, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_u16, u64, u16, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_u32, u64, u32, Bitor, NormalOutPromote);
DEFINE_BINARY_KERNEL(bitor_u64_u64, u64, u64, Bitor, NormalOutPromote);