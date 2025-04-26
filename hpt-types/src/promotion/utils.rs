/// this macro is used to implement the FloatOutBinaryPromote trait for a given type
#[macro_export]
macro_rules! impl_float_out_binary_promote {
    ($lhs:ty, $rhs:ty, $output:ty, $intermediate:ty) => {
        impl FloatOutBinaryPromote<$rhs> for $lhs {
            type Output = $output;
            type Intermediate = $intermediate;
        }
        #[cfg(feature = "cuda")]
        impl FloatOutBinaryPromote<Scalar<$rhs>> for Scalar<$lhs> {
            type Output = Scalar<$output>;
            type Intermediate = Scalar<$intermediate>;
        }
        paste::paste! {
            impl FloatOutBinaryPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
                type Intermediate = [<$intermediate _promote>];
            }
        }
    };
}

/// this macro is used to implement the FloatOutBinaryPromote trait for a given type
#[macro_export]
macro_rules! impl_normal_out_promote {
    ($lhs:ty, $rhs:ty, $output:ty, $intermediate:ty) => {
        impl NormalOutPromote<$rhs> for $lhs {
            type Output = $output;
            type Intermediate = $intermediate;
        }
        #[cfg(feature = "cuda")]
        impl NormalOutPromote<Scalar<$rhs>> for Scalar<$lhs> {
            type Output = Scalar<$output>;
            type Intermediate = Scalar<$intermediate>;
        }
        paste::paste! {
            impl NormalOutPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
                type Intermediate = [<$intermediate _promote>];
            }
        }
    };
}

/// this macro is used to implement the SimdCmpPromote trait for a given type
#[macro_export]
macro_rules! impl_simd_cmp_promote {
    ($lhs:ty, $rhs:ty, $output:ty) => {
        paste::paste! {
            impl SimdCmpPromote<[<$rhs _promote>]> for [<$lhs _promote>] {
                type Output = [<$output _promote>];
            }
        }
    };
}

/// this macro is used to implement the FloatOutUnaryPromote trait for a given type
#[macro_export]
macro_rules! impl_float_out_unary_promote {
    ($lhs:ty, $output:ty, $intermediate:ty) => {
        impl FloatOutUnaryPromote for $lhs {
            type Output = $output;
            type Intermediate = $intermediate;
        }
        #[cfg(feature = "cuda")]
        impl FloatOutUnaryPromote for Scalar<$lhs> {
            type Output = Scalar<$output>;
            type Intermediate = Scalar<$intermediate>;
        }
        paste::paste! {
            impl FloatOutUnaryPromote for [<$lhs _promote>] {
                type Output = [<$output _promote>];
                type Intermediate = [<$intermediate _promote>];
            }
        }
    };
}

use crate::dtype::DType;
use crate::dtype::ToDType;
use crate::type_promote::{FloatOutBinaryPromote, FloatOutUnaryPromote, NormalOutPromote};

#[allow(missing_docs)]
pub fn promote_float_unary(lhs: DType) -> DType {
    match lhs {
        DType::Bool => <bool as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::I8 => <i8 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::U8 => <u8 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::I16 => <i16 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::U16 => <u16 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::I32 => <i32 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::U32 => <u32 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::I64 => unreachable!(),
        DType::F32 => <f32 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::F16 => <half::f16 as FloatOutUnaryPromote>::Output::to_dtype(),
        DType::BF16 => <half::bf16 as FloatOutUnaryPromote>::Output::to_dtype(),
    }
}

#[allow(missing_docs)]
#[duplicate::duplicate_item(
    func_name               trait_name;
    [promote_normal_binary] [NormalOutPromote];
    [promote_float_binary]  [FloatOutBinaryPromote];
)]
pub fn func_name(lhs: DType, rhs: DType) -> DType {
    match (lhs, rhs) {
        (DType::Bool, DType::Bool) => <bool as trait_name<bool>>::Output::to_dtype(),
        (DType::Bool, DType::I8) => <bool as trait_name<i8>>::Output::to_dtype(),
        (DType::Bool, DType::U8) => <bool as trait_name<u8>>::Output::to_dtype(),
        (DType::Bool, DType::I16) => <bool as trait_name<i16>>::Output::to_dtype(),
        (DType::Bool, DType::U16) => <bool as trait_name<u16>>::Output::to_dtype(),
        (DType::Bool, DType::I32) => <bool as trait_name<i32>>::Output::to_dtype(),
        (DType::Bool, DType::U32) => <bool as trait_name<u32>>::Output::to_dtype(),
        (DType::Bool, DType::I64) => unreachable!(),
        (DType::Bool, DType::F32) => <bool as trait_name<f32>>::Output::to_dtype(),
        (DType::Bool, DType::F16) => <bool as trait_name<half::f16>>::Output::to_dtype(),
        (DType::Bool, DType::BF16) => <bool as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::I8, DType::Bool) => <i8 as trait_name<bool>>::Output::to_dtype(),
        (DType::I8, DType::I8) => <i8 as trait_name<i8>>::Output::to_dtype(),
        (DType::I8, DType::U8) => <i8 as trait_name<u8>>::Output::to_dtype(),
        (DType::I8, DType::I16) => <i8 as trait_name<i16>>::Output::to_dtype(),
        (DType::I8, DType::U16) => <i8 as trait_name<u16>>::Output::to_dtype(),
        (DType::I8, DType::I32) => <i8 as trait_name<i32>>::Output::to_dtype(),
        (DType::I8, DType::U32) => <i8 as trait_name<u32>>::Output::to_dtype(),
        (DType::I8, DType::I64) => unreachable!(),
        (DType::I8, DType::F32) => <i8 as trait_name<f32>>::Output::to_dtype(),
        (DType::I8, DType::F16) => <i8 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::I8, DType::BF16) => <i8 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::U8, DType::Bool) => <u8 as trait_name<bool>>::Output::to_dtype(),
        (DType::U8, DType::I8) => <u8 as trait_name<i8>>::Output::to_dtype(),
        (DType::U8, DType::U8) => <u8 as trait_name<u8>>::Output::to_dtype(),
        (DType::U8, DType::I16) => <u8 as trait_name<i16>>::Output::to_dtype(),
        (DType::U8, DType::U16) => <u8 as trait_name<u16>>::Output::to_dtype(),
        (DType::U8, DType::I32) => <u8 as trait_name<i32>>::Output::to_dtype(),
        (DType::U8, DType::U32) => <u8 as trait_name<u32>>::Output::to_dtype(),
        (DType::U8, DType::I64) => unreachable!(),
        (DType::U8, DType::F32) => <u8 as trait_name<f32>>::Output::to_dtype(),
        (DType::U8, DType::F16) => <u8 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::U8, DType::BF16) => <u8 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::I16, DType::Bool) => <i16 as trait_name<bool>>::Output::to_dtype(),
        (DType::I16, DType::I8) => <i16 as trait_name<i8>>::Output::to_dtype(),
        (DType::I16, DType::U8) => <i16 as trait_name<u8>>::Output::to_dtype(),
        (DType::I16, DType::I16) => <i16 as trait_name<i16>>::Output::to_dtype(),
        (DType::I16, DType::U16) => <i16 as trait_name<u16>>::Output::to_dtype(),
        (DType::I16, DType::I32) => <i16 as trait_name<i32>>::Output::to_dtype(),
        (DType::I16, DType::U32) => <i16 as trait_name<u32>>::Output::to_dtype(),
        (DType::I16, DType::I64) => unreachable!(),
        (DType::I16, DType::F32) => <i16 as trait_name<f32>>::Output::to_dtype(),
        (DType::I16, DType::F16) => <i16 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::I16, DType::BF16) => <i16 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::U16, DType::Bool) => <u16 as trait_name<bool>>::Output::to_dtype(),
        (DType::U16, DType::I8) => <u16 as trait_name<i8>>::Output::to_dtype(),
        (DType::U16, DType::U8) => <u16 as trait_name<u8>>::Output::to_dtype(),
        (DType::U16, DType::I16) => <u16 as trait_name<i16>>::Output::to_dtype(),
        (DType::U16, DType::U16) => <u16 as trait_name<u16>>::Output::to_dtype(),
        (DType::U16, DType::I32) => <u16 as trait_name<i32>>::Output::to_dtype(),
        (DType::U16, DType::U32) => <u16 as trait_name<u32>>::Output::to_dtype(),
        (DType::U16, DType::I64) => unreachable!(),
        (DType::U16, DType::F32) => <u16 as trait_name<f32>>::Output::to_dtype(),
        (DType::U16, DType::F16) => <u16 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::U16, DType::BF16) => <u16 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::I32, DType::Bool) => <i32 as trait_name<bool>>::Output::to_dtype(),
        (DType::I32, DType::I8) => <i32 as trait_name<i8>>::Output::to_dtype(),
        (DType::I32, DType::U8) => <i32 as trait_name<u8>>::Output::to_dtype(),
        (DType::I32, DType::I16) => <i32 as trait_name<i16>>::Output::to_dtype(),
        (DType::I32, DType::U16) => <i32 as trait_name<u16>>::Output::to_dtype(),
        (DType::I32, DType::I32) => <i32 as trait_name<i32>>::Output::to_dtype(),
        (DType::I32, DType::U32) => <i32 as trait_name<u32>>::Output::to_dtype(),
        (DType::I32, DType::I64) => unreachable!(),
        (DType::I32, DType::F32) => <i32 as trait_name<f32>>::Output::to_dtype(),
        (DType::I32, DType::F16) => <i32 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::I32, DType::BF16) => <i32 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::U32, DType::Bool) => <u32 as trait_name<bool>>::Output::to_dtype(),
        (DType::U32, DType::I8) => <u32 as trait_name<i8>>::Output::to_dtype(),
        (DType::U32, DType::U8) => <u32 as trait_name<u8>>::Output::to_dtype(),
        (DType::U32, DType::I16) => <u32 as trait_name<i16>>::Output::to_dtype(),
        (DType::U32, DType::U16) => <u32 as trait_name<u16>>::Output::to_dtype(),
        (DType::U32, DType::I32) => <u32 as trait_name<i32>>::Output::to_dtype(),
        (DType::U32, DType::U32) => <u32 as trait_name<u32>>::Output::to_dtype(),
        (DType::U32, DType::I64) => unreachable!(),
        (DType::U32, DType::F32) => <u32 as trait_name<f32>>::Output::to_dtype(),
        (DType::U32, DType::F16) => <u32 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::U32, DType::BF16) => <u32 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::I64, DType::Bool) => unreachable!(),
        (DType::I64, DType::I8) => unreachable!(),
        (DType::I64, DType::U8) => unreachable!(),
        (DType::I64, DType::I16) => unreachable!(),
        (DType::I64, DType::U16) => unreachable!(),
        (DType::I64, DType::I32) => unreachable!(),
        (DType::I64, DType::U32) => unreachable!(),
        (DType::I64, DType::I64) => unreachable!(),
        (DType::I64, DType::F32) => unreachable!(),
        (DType::I64, DType::F16) => unreachable!(),
        (DType::I64, DType::BF16) => unreachable!(),
        (DType::F32, DType::Bool) => <f32 as trait_name<bool>>::Output::to_dtype(),
        (DType::F32, DType::I8) => <f32 as trait_name<i8>>::Output::to_dtype(),
        (DType::F32, DType::U8) => <f32 as trait_name<u8>>::Output::to_dtype(),
        (DType::F32, DType::I16) => <f32 as trait_name<i16>>::Output::to_dtype(),
        (DType::F32, DType::U16) => <f32 as trait_name<u16>>::Output::to_dtype(),
        (DType::F32, DType::I32) => <f32 as trait_name<i32>>::Output::to_dtype(),
        (DType::F32, DType::U32) => <f32 as trait_name<u32>>::Output::to_dtype(),
        (DType::F32, DType::I64) => unreachable!(),
        (DType::F32, DType::F32) => <f32 as trait_name<f32>>::Output::to_dtype(),
        (DType::F32, DType::F16) => <f32 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::F32, DType::BF16) => <f32 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::F16, DType::Bool) => <half::f16 as trait_name<bool>>::Output::to_dtype(),
        (DType::F16, DType::I8) => <half::f16 as trait_name<i8>>::Output::to_dtype(),
        (DType::F16, DType::U8) => <half::f16 as trait_name<u8>>::Output::to_dtype(),
        (DType::F16, DType::I16) => <half::f16 as trait_name<i16>>::Output::to_dtype(),
        (DType::F16, DType::U16) => <half::f16 as trait_name<u16>>::Output::to_dtype(),
        (DType::F16, DType::I32) => <half::f16 as trait_name<i32>>::Output::to_dtype(),
        (DType::F16, DType::U32) => <half::f16 as trait_name<u32>>::Output::to_dtype(),
        (DType::F16, DType::I64) => unreachable!(),
        (DType::F16, DType::F32) => <half::f16 as trait_name<f32>>::Output::to_dtype(),
        (DType::F16, DType::F16) => <half::f16 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::F16, DType::BF16) => <half::f16 as trait_name<half::bf16>>::Output::to_dtype(),
        (DType::BF16, DType::Bool) => <half::bf16 as trait_name<bool>>::Output::to_dtype(),
        (DType::BF16, DType::I8) => <half::bf16 as trait_name<i8>>::Output::to_dtype(),
        (DType::BF16, DType::U8) => <half::bf16 as trait_name<u8>>::Output::to_dtype(),
        (DType::BF16, DType::I16) => <half::bf16 as trait_name<i16>>::Output::to_dtype(),
        (DType::BF16, DType::U16) => <half::bf16 as trait_name<u16>>::Output::to_dtype(),
        (DType::BF16, DType::I32) => <half::bf16 as trait_name<i32>>::Output::to_dtype(),
        (DType::BF16, DType::U32) => <half::bf16 as trait_name<u32>>::Output::to_dtype(),
        (DType::BF16, DType::I64) => unreachable!(),
        (DType::BF16, DType::F32) => <half::bf16 as trait_name<f32>>::Output::to_dtype(),
        (DType::BF16, DType::F16) => <half::bf16 as trait_name<half::f16>>::Output::to_dtype(),
        (DType::BF16, DType::BF16) => <half::bf16 as trait_name<half::bf16>>::Output::to_dtype(),
    }
}
