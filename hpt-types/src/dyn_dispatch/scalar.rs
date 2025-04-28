#![allow(missing_docs)]

use std::sync::Arc;

use hpt_macros::impl_dispatch;

use crate::dtype::DType;
use crate::into_scalar::Cast;
use crate::type_promote::FloatOutUnary;
use crate::type_promote::NormalOutPromote;
use crate::type_promote::FloatOutUnaryPromote;
use crate::type_promote::FloatOutBinaryPromote;
use crate::type_promote::NormalOut;
use crate::type_promote::FloatOutBinary;
use crate::type_promote::NormalOutUnary;

use half::{bf16, f16};

type Fn3Type = fn(usize, usize, usize, usize);
type Fn2Type = fn(usize, usize, usize);
type Fn1Type = fn(usize, usize);

pub fn dispatch_fill(lhs: DType, val: f64) -> Arc<dyn Fn(usize) + Send + Sync> {
    macro_rules! fill_arm {
        ($lhs:ident, $cast:expr) => {{
            let val = $cast;
            Arc::new(move |res: usize| {
                let ptr = res as *mut $lhs;
                unsafe { *ptr = val };
            })
        }};
    }
    match lhs {
        DType::Bool => {
            let val = val != 0.0;
            Arc::new(move |res: usize| {
                let ptr = res as *mut bool;
                unsafe { *ptr = val };
            })
        }
        DType::I8 => fill_arm!(i8, val as i8),
        DType::U8 => fill_arm!(u8, val as u8),
        DType::I16 => fill_arm!(i16, val as i16),
        DType::U16 => fill_arm!(u16, val as u16),
        DType::I32 => fill_arm!(i32, val as i32),
        DType::U32 => fill_arm!(u32, val as u32),
        DType::I64 => fill_arm!(i64, val as i64),
        DType::F32 => fill_arm!(f32, val as f32),
        DType::F16 => fill_arm!(f16, half::f16::from_f64(val)),
        DType::BF16 => fill_arm!(bf16, half::bf16::from_f64(val)),
    }
}

pub fn dispatch_arange(lhs: DType, start: f64) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arange_arm {
        ($lhs:ident, $start_cast:expr) => {{
            let start = $start_cast;
            Arc::new(move |res: usize, idx: usize| {
                let ptr = res as *mut $lhs;
                unsafe { *ptr = start._add(idx) as $lhs };
            })
        }};
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => arange_arm!(i8, start as i8),
        DType::U8 => arange_arm!(u8, start as u8),
        DType::I16 => arange_arm!(i16, start as i16),
        DType::U16 => arange_arm!(u16, start as u16),
        DType::I32 => arange_arm!(i32, start as i32),
        DType::U32 => arange_arm!(u32, start as u32),
        DType::I64 => arange_arm!(i64, start as i64),
        DType::F32 => arange_arm!(f32, start as f32),
        DType::F16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut f16;
            unsafe { *ptr = start._add(idx).cast() };
        }),
        DType::BF16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut bf16;
            unsafe { *ptr = start._add(idx).cast() };
        }),
    }
}

pub fn dispatch_arange_step(
    lhs: DType,
    start: f64,
    step: f64,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arange_arm {
        ($lhs:ident, $start_cast:expr, $step_cast:expr) => {{
            let start = $start_cast;
            let step = $step_cast;
            Arc::new(move |res: usize, idx: usize| {
                let ptr = res as *mut $lhs;
                unsafe { *ptr = start._add(idx._mul(step)) as $lhs };
            })
        }};
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => arange_arm!(i8, start as i8, step as i8),
        DType::U8 => arange_arm!(u8, start as u8, step as u8),
        DType::I16 => arange_arm!(i16, start as i16, step as i16),
        DType::U16 => arange_arm!(u16, start as u16, step as u16),
        DType::I32 => arange_arm!(i32, start as i32, step as i32),
        DType::U32 => arange_arm!(u32, start as u32, step as u32),
        DType::I64 => arange_arm!(i64, start as i64, step as i64),
        DType::F32 => arange_arm!(f32, start as f32, step as f32),
        DType::F16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut f16;
            unsafe { *ptr = start._add(idx._mul(step)).cast() };
        }),
        DType::BF16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut bf16;
            unsafe { *ptr = start._add(idx._mul(step)).cast() };
        }),
    }
}

pub fn dispatch_eye(
    lhs: DType,
    m: usize,
    k: usize,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident, $one:expr, $zero:expr) => {{
            Arc::new(move |res: usize, idx: usize| {
                let ptr = res as *mut $lhs;
                let row = idx / m;
                let col = idx % m;
                if col == row + k {
                    unsafe { *ptr = $one };
                } else {
                    unsafe { *ptr = $zero };
                }
            })
        }};
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => {
            arm!(i8, 1i8, 0i8)
        }
        DType::U8 => arm!(u8, 1u8, 0u8),
        DType::I16 => arm!(i16, 1i16, 0i16),
        DType::U16 => arm!(u16, 1u16, 0u16),
        DType::I32 => arm!(i32, 1i32, 0i32),
        DType::U32 => arm!(u32, 1u32, 0u32),
        DType::I64 => arm!(i64, 1i64, 0i64),
        DType::F32 => arm!(f32, 1.0f32, 0.0f32),
        DType::F16 => arm!(f16, f16::from_f32_const(1.0), f16::from_f32_const(0.0)),
        DType::BF16 => arm!(bf16, bf16::from_f32_const(1.0), bf16::from_f32_const(0.0)),
    }
}

pub fn dispatch_linspace(
    lhs: DType,
    start: f64,
    end: f64,
    num: usize,
    step: f64,
    include_end: bool,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident) => {{
            Arc::new(move |res: usize, idx: usize| {
                let ptr = res as *mut $lhs;
                if include_end && idx == num - 1 {
                    unsafe { *ptr = end as $lhs };
                } else {
                    unsafe { *ptr = start._add(idx._mul(step)) as $lhs };
                }
            })
        }};
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => {
            arm!(i8)
        }
        DType::U8 => arm!(u8),
        DType::I16 => arm!(i16),
        DType::U16 => arm!(u16),
        DType::I32 => arm!(i32),
        DType::U32 => arm!(u32),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut f16;
            if include_end && idx == num - 1 {
                unsafe { *ptr = half::f16::from_f64(end) };
            } else {
                unsafe { *ptr = half::f16::from_f64(start._add(idx._mul(step))) };
            }
        }),
        DType::BF16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut bf16;
            if include_end && idx == num - 1 {
                unsafe { *ptr = half::bf16::from_f64(end) };
            } else {
                unsafe { *ptr = half::bf16::from_f64(start._add(idx._mul(step))) };
            }
        }),
    }
}

pub fn dispatch_logspace(
    lhs: DType,
    base: f64,
    start: f64,
    step: f64,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident) => {
            Arc::new(move |res: usize, idx: usize| {
                let ptr = res as *mut $lhs;
                unsafe { *ptr = base._pow(start._add(idx._mul(step))) as $lhs };
            })
        };
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => {
            arm!(i8)
        }
        DType::U8 => arm!(u8),
        DType::I16 => arm!(i16),
        DType::U16 => arm!(u16),
        DType::I32 => arm!(i32),
        DType::U32 => arm!(u32),
        DType::I64 => arm!(i64),
        DType::F32 => arm!(f32),
        DType::F16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut f16;
            unsafe { *ptr = base._pow(start._add(idx._mul(step))).cast() };
        }),
        DType::BF16 => Arc::new(move |res: usize, idx: usize| {
            let ptr = res as *mut bf16;
            unsafe { *ptr = base._pow(start._add(idx._mul(step))).cast() };
        }),
    }
}

pub fn dispatch_geomspace(
    lhs: DType,
    start: f64,
    step: f64,
    both_negative: bool,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident, $ten: expr) => {
            if both_negative {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut $lhs;
                    let val = $ten._pow(start._add(idx._mul(step)))._neg() as $lhs;
                    unsafe { *ptr = val };
                })
            } else {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut $lhs;
                    let val = $ten._pow(start._add(idx._mul(step))) as $lhs;
                    unsafe { *ptr = val };
                })
            }
        };
    }
    match lhs {
        DType::Bool => {
            unimplemented!("Bool type is not supported for arange");
        }
        DType::I8 => {
            arm!(i8, 10.0f64)
        }
        DType::U8 => arm!(u8, 10.0f64),
        DType::I16 => arm!(i16, 10.0f64),
        DType::U16 => arm!(u16, 10.0f64),
        DType::I32 => arm!(i32, 10.0f64),
        DType::U32 => arm!(u32, 10.0f64),
        DType::I64 => arm!(i64, 10.0f64),
        DType::F32 => arm!(f32, 10.0f64),
        DType::F16 => {
            if both_negative {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut f16;
                    let val = 10.0f64._pow(start._add(idx._mul(step)))._neg().cast();
                    unsafe { *ptr = val };
                })
            } else {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut f16;
                    let val = 10.0f64._pow(start._add(idx._mul(step))).cast();
                    unsafe { *ptr = val };
                })
            }
        }
        DType::BF16 => {
            if both_negative {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut bf16;
                    let val = 10.0f64._pow(start._add(idx._mul(step)))._neg().cast();
                    unsafe { *ptr = val };
                })
            } else {
                Arc::new(move |res: usize, idx: usize| {
                    let ptr = res as *mut bf16;
                    let val = 10.0f64._pow(start._add(idx._mul(step))).cast();
                    unsafe { *ptr = val };
                })
            }
        }
    }
}

pub fn dispatch_div(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(FloatOutBinaryPromote, FloatOutBinary, _div, true, 2, 1)
}

pub fn dispatch_log(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(FloatOutBinaryPromote, FloatOutBinary, _log, true, 2, 1)
}

pub fn dispatch_hypot(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(FloatOutBinaryPromote, FloatOutBinary, _hypot, true, 2, 1)
}

pub fn dispatch_pow(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(FloatOutBinaryPromote, FloatOutBinary, _pow, true, 2, 1)
}

pub fn dispatch_add(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _add, true, 2, 1)
}

pub fn dispatch_sub(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _sub, true, 2, 1)
}

pub fn dispatch_mul_add(a: DType, b: DType, c: DType) -> Fn3Type {
    assert_eq!(a, b);
    assert_eq!(a, c);
    impl_dispatch!(NormalOutPromote, NormalOut, _mul_add, false, 3, 1)
}

pub fn dispatch_mul(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _mul, true, 2, 1)
}

pub fn dispatch_rem(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _rem, true, 2, 1)
}

pub fn dispatch_max(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _max, true, 2, 1)
}

pub fn dispatch_min(lhs: DType, rhs: DType) -> Fn2Type {
    impl_dispatch!(NormalOutPromote, NormalOut, _min, true, 2, 1)
}

pub fn dispatch_clamp(a: DType, b: DType, c: DType) -> Fn3Type {
    assert_eq!(a, b);
    assert_eq!(a, c);
    impl_dispatch!(NormalOutPromote, NormalOut, _clamp, false, 3, 1)
}

pub fn dispatch_square(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _square, true, 1, 1)
}

pub fn dispatch_abs(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _abs, true, 1, 1)
}

pub fn dispatch_ceil(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _ceil, true, 1, 1)
}

pub fn dispatch_floor(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _floor, true, 1, 1)
}

pub fn dispatch_neg(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _neg, true, 1, 1)
}

pub fn dispatch_round(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _round, true, 1, 1)
}

pub fn dispatch_signum(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _signum, true, 1, 1)
}

pub fn dispatch_trunc(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _trunc, true, 1, 1)
}

pub fn dispatch_leaky_relu(lhs: DType, rhs: DType) -> Fn2Type {
    assert_eq!(lhs, rhs);
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _leaky_relu, false, 2, 1)
}

pub fn dispatch_relu(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _relu, true, 1, 1)
}

pub fn dispatch_relu6(lhs: DType) -> Fn1Type {
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _relu6, true, 1, 1)
}

pub fn dispatch_copysign(lhs: DType, rhs: DType) -> Fn2Type {
    assert_eq!(lhs, rhs);
    impl_dispatch!(NormalOutPromote, NormalOutUnary, _copysign, false, 2, 1)
}

#[duplicate::duplicate_item(
    func_name           method;
    [dispatch_celu]     [_celu];
    [dispatch_elu]      [_elu];
)]
pub fn func_name(lhs: DType, alpha: f64) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident) => {{
            type Output = <$lhs as FloatOutUnaryPromote>::Output;
            let alpha: Output = alpha.cast();
            Arc::new(move |lhs: usize, res: usize| {
                let ptr = res as *mut Output;
                let lhs = lhs as *const $lhs;
                unsafe { *ptr = (*lhs).method(alpha) };
            })
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
    func_name           method;
    [dispatch_selu]     [_selu];
)]
pub fn func_name(
    lhs: DType,
    arg1: f64,
    arg2: f64,
) -> Arc<dyn Fn(usize, usize) + Send + Sync> {
    macro_rules! arm {
        ($lhs:ident) => {{
            type Output = <$lhs as FloatOutUnaryPromote>::Output;
            let arg1: Output = arg1.cast();
            let arg2: Output = arg2.cast();
            Arc::new(move |lhs: usize, res: usize| {
                let ptr = res as *mut Output;
                let lhs = lhs as *const $lhs;
                unsafe { *ptr = (*lhs).method(arg1, arg2) };
            })
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

pub fn dispatch_copy(lhs: DType) -> fn(usize, usize) {
    macro_rules! arm {
        ($lhs:ident) => {{
            |lhs: usize, res: usize| {
                let ptr = res as *mut $lhs;
                let lhs = lhs as *const $lhs;
                unsafe { *ptr = *lhs };
            }
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
    func_name               method;
    [dispatch_sin]          [_sin];
    [dispatch_cos]          [_cos];
    [dispatch_tan]          [_tan];
    [dispatch_asin]         [_asin];
    [dispatch_acos]         [_acos];
    [dispatch_atan]         [_atan];
    [dispatch_sinh]         [_sinh];
    [dispatch_cosh]         [_cosh];
    [dispatch_tanh]         [_tanh];
    [dispatch_asinh]        [_asinh];
    [dispatch_acosh]        [_acosh];
    [dispatch_atanh]        [_atanh];
    [dispatch_exp]          [_exp];
    [dispatch_exp2]         [_exp2];
    [dispatch_expm1]        [_expm1];
    [dispatch_ln]           [_ln];
    [dispatch_log1p]        [_log1p];
    [dispatch_log2]         [_log2];
    [dispatch_log10]        [_log10];
    [dispatch_sqrt]         [_sqrt];
    [dispatch_cbrt]         [_cbrt];
    [dispatch_recip]        [_recip];
    [dispatch_erf]          [_erf];
    [dispatch_sigmoid]      [_sigmoid];
    [dispatch_gelu]         [_gelu];
    [dispatch_hard_sigmoid] [_hard_sigmoid];
    [dispatch_hard_swish]  [_hard_swish];
    [dispatch_softplus]    [_softplus];
    [dispatch_softsign]    [_softsign];
    [dispatch_mish]        [_mish];
)]
pub fn func_name(lhs: DType) -> Fn1Type {
    impl_dispatch!(FloatOutUnaryPromote, FloatOutUnary, method, true, 1, 1)
}
