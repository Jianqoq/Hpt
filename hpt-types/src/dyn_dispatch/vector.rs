#![allow(missing_docs)]

use crate::dtype::DType;
use crate::dtype::TypeCommon;
use hpt_macros::impl_dispatch_simd;

use crate::type_promote::NormalOut;
use crate::type_promote::NormalOutPromote;

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

type Fn2Type = fn(usize, usize, usize);

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
