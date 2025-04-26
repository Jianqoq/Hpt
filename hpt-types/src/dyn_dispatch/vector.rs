#![allow(missing_docs)]

use hpt_macros::impl_dispatch_simd;
use std::sync::Arc;
use crate::dtype::TypeCommon;
use crate::into_scalar::Cast;
use crate::vectors::traits::VecTrait;
use crate::dtype::DType;

use crate::type_promote::FloatOutUnary;
use crate::type_promote::NormalOutPromote;
use crate::type_promote::NormalOut;
use crate::type_promote::FloatOutUnaryPromote;

#[cfg(any(
    all(not(target_feature = "avx2"), target_feature = "sse"),
    target_arch = "arm",
    target_arch = "aarch64",
    target_feature = "neon"
))]
use crate::vectors::arch_simd::_128bit::*;
#[cfg(target_feature = "avx2")]
use crate::vectors::arch_simd::_256bit::*;

use half::{bf16, f16};

type Fn4Type = fn(usize, usize, usize, usize, usize);
type Fn3Type = fn(usize, usize, usize, usize);
type Fn2Type = fn(usize, usize, usize);
type Fn1Type = fn(usize, usize);

#[duplicate::duplicate_item(
    func_name                   method        unroll;
    [dispatch_simd_sin]         [_sin]        [1];
    [dispatch_simd_cos]         [_cos]        [1];
    [dispatch_simd_tan]         [_tan]        [1];
    [dispatch_simd_asin]        [_asin]       [1];
    [dispatch_simd_acos]        [_acos]       [1];
    [dispatch_simd_atan]        [_atan]       [1];
    [dispatch_simd_sinh]        [_sinh]       [1];
    [dispatch_simd_cosh]        [_cosh]       [1];
    [dispatch_simd_tanh]        [_tanh]       [1];
    [dispatch_simd_asinh]       [_asinh]      [1];
    [dispatch_simd_acosh]       [_acosh]      [1];
    [dispatch_simd_atanh]       [_atanh]      [1];
    [dispatch_simd_exp]         [_exp]        [1];
    [dispatch_simd_exp2]        [_exp2]       [1];
    [dispatch_simd_expm1]       [_expm1]      [1];
    [dispatch_simd_ln]          [_ln]         [1];
    [dispatch_simd_log1p]       [_log1p]      [1];
    [dispatch_simd_log2]        [_log2]       [1];
    [dispatch_simd_log10]       [_log10]      [1];
    [dispatch_simd_sqrt]        [_sqrt]       [1];
    [dispatch_simd_cbrt]        [_cbrt]       [1];
    [dispatch_simd_recip]       [_recip]      [1];
    [dispatch_simd_erf]         [_erf]        [1];
    [dispatch_simd_sigmoid]     [_sigmoid]    [4];
    [dispatch_simd_gelu]        [_gelu]       [1];
    [dispatch_simd_hard_sigmoid] [_hard_sigmoid] [1];
    [dispatch_simd_hard_swish]  [_hard_swish]  [1];
    [dispatch_simd_softplus]    [_softplus]    [1];
    [dispatch_simd_softsign]    [_softsign]    [1];
    [dispatch_simd_mish]        [_mish]        [1];
)]
pub fn func_name(lhs: DType) -> (Fn1Type, usize) {
    (
        impl_dispatch_simd!(FloatOutUnaryPromote, FloatOutUnary, method, true, 1, unroll),
        unroll,
    )
}

#[duplicate::duplicate_item(
    func_name   method;
    [dispatch_simd_celu]    [_celu];
    [dispatch_simd_elu]     [_elu];
)]
pub fn func_name(
    lhs: DType,
    alpha: f64,
) -> (Arc<dyn Fn(usize, usize) + Send + Sync>, usize) {
    macro_rules! arm {
        ($lhs:ident) => {{
            type VecType = <$lhs as TypeCommon>::Vec;
            type Output = <VecType as FloatOutUnaryPromote>::Output;
            let alpha = Output::splat(alpha.cast());
            (
                Arc::new(move |lhs: usize, res: usize| {
                    let ptr = res as *mut Output;
                    let lhs = lhs as *const VecType;
                    unsafe {
                        let a0 = lhs.read_unaligned();
                        let a1 = lhs.add(1).read_unaligned();
                        ptr.write_unaligned(a0.method(alpha));
                        ptr.add(1).write_unaligned(a1.method(alpha));
                    };
                }),
                2,
            )
        }};
    }
    match lhs {
        DType::Bool => arm!(bool),
        DType::I8 => arm!(i8),
        DType::U8 => arm!(u8),
        DType::I16 => arm!(i16),
        DType::U16 => arm!(u16),
        DType::I32 => arm!(i32),
        DType::U32 => arm!(u32),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F16 => arm!(f16),
        DType::BF16 => arm!(bf16),
    }
}

#[duplicate::duplicate_item(
    func_name   method;
    [dispatch_simd_selu]    [_selu];
)]
pub fn func_name(
    lhs: DType,
    arg1: f64,
    arg2: f64,
) -> (Arc<dyn Fn(usize, usize) + Send + Sync>, usize) {
    macro_rules! arm {
        ($lhs:ident) => {{
            type VecType = <$lhs as TypeCommon>::Vec;
            type Output = <VecType as FloatOutUnaryPromote>::Output;
            let arg1 = Output::splat(arg1.cast());
            let arg2 = Output::splat(arg2.cast());
            (
                Arc::new(move |lhs: usize, res: usize| {
                    let ptr = res as *mut Output;
                    let lhs = lhs as *const VecType;
                    unsafe {
                        let a0 = lhs.read_unaligned();
                        let a1 = lhs.add(1).read_unaligned();
                        ptr.write_unaligned(a0.method(arg1, arg2));
                        ptr.add(1).write_unaligned(a1.method(arg1, arg2));
                    };
                }),
                2,
            )
        }};
    }
    match lhs {
        DType::Bool => arm!(bool),
        DType::I8 => arm!(i8),
        DType::U8 => arm!(u8),
        DType::I16 => arm!(i16),
        DType::U16 => arm!(u16),
        DType::I32 => arm!(i32),
        DType::U32 => arm!(u32),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F16 => arm!(f16),
        DType::BF16 => arm!(bf16),
    }
}

pub fn dispatch_simd_copy(lhs: DType) -> (fn(usize, usize), usize) {
    macro_rules! arm {
        ($lhs:ident) => {{
            type VecType = <$lhs as TypeCommon>::Vec;
            (
                |lhs: usize, res: usize| {
                    let ptr = res as *mut VecType;
                    let lhs = lhs as *const VecType;
                    unsafe {
                        let a0 = lhs.read_unaligned();
                        let a1 = lhs.add(1).read_unaligned();
                        ptr.write_unaligned(a0);
                        ptr.add(1).write_unaligned(a1);
                    };
                },
                2,
            )
        }};
    }
    match lhs {
        DType::Bool => arm!(bool),
        DType::I8 => arm!(i8),
        DType::U8 => arm!(u8),
        DType::I16 => arm!(i16),
        DType::U16 => arm!(u16),
        DType::I32 => arm!(i32),
        DType::U32 => arm!(u32),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F16 => arm!(f16),
        DType::BF16 => arm!(bf16),
    }
}

#[duplicate::duplicate_item(
    func_name               method       unroll;
    [dispatch_simd_add]     [_add]        [2];
    [dispatch_simd_sub]     [_sub]        [2];
    [dispatch_simd_mul]     [_mul]        [2];
    [dispatch_simd_rem]     [_rem]        [2];
    [dispatch_simd_max]     [_max]        [2];
    [dispatch_simd_min]     [_min]        [2];
)]
pub fn func_name(lhs: DType, rhs: DType) -> (Fn2Type, usize) {
    (
        impl_dispatch_simd!(NormalOutPromote, NormalOut, method, true, 2, unroll),
        unroll,
    )
}
