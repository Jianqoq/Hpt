use rand::distributions::uniform::SampleUniform;

use crate::{dtype::TypeCommon, traits::VecTrait};

type F32Vec = <f32 as TypeCommon>::Vec;
type F16Vec = <half::f16 as TypeCommon>::Vec;
type Bf16Vec = <half::bf16 as TypeCommon>::Vec;
type F64Vec = <f64 as TypeCommon>::Vec;
type I8Vec = <i8 as TypeCommon>::Vec;
type I16Vec = <i16 as TypeCommon>::Vec;
type I32Vec = <i32 as TypeCommon>::Vec;
type I64Vec = <i64 as TypeCommon>::Vec;
type U8Vec = <u8 as TypeCommon>::Vec;
type U16Vec = <u16 as TypeCommon>::Vec;
type U32Vec = <u32 as TypeCommon>::Vec;
type U64Vec = <u64 as TypeCommon>::Vec;
type IsizeVec = <isize as TypeCommon>::Vec;
type UsizeVec = <usize as TypeCommon>::Vec;

pub(crate) fn f32_to_f16<const N: usize, F>(ptr: [f32; N], conv_func: F, msg: &str)
where
    F: Fn([f32; N]) -> [half::f16; N],
{
    let mut result = [half::f16::from_f32_const(0.0); N];
    for i in 0..N {
        result[i] = half::f16::from_f32_const(ptr[i]);
    }
    let simd_result = conv_func(ptr);
    for i in 0..N {
        if result[i] != simd_result[i] && (!result[i].is_nan() && !simd_result[i].is_nan()) {
            panic!("{}: actual: {} != simd: {}", msg, result[i], simd_result[i]);
        }
    }
}

pub(crate) fn f16_to_f32<const N: usize, F>(ptr: [half::f16; N], conv_func: F, msg: &str)
where
    F: Fn([half::f16; N]) -> [f32; N],
{
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = ptr[i].to_f32();
    }
    let simd_result = conv_func(ptr);
    for i in 0..N {
        if result[i] != simd_result[i] && (!result[i].is_nan() && !simd_result[i].is_nan()) {
            panic!("{}: actual: {} != simd: {}", msg, result[i], simd_result[i]);
        }
    }
}

pub(crate) fn f32_to_bf16<const N: usize, F>(ptr: [f32; N], conv_func: F, msg: &str)
where
    F: Fn([f32; N]) -> [half::bf16; N],
{
    let mut result = [half::bf16::from_f32_const(0.0); N];
    for i in 0..N {
        result[i] = half::bf16::from_f32_const(ptr[i]);
    }
    let simd_result = conv_func(ptr);
    for i in 0..N {
        if result[i] != simd_result[i] && (!result[i].is_nan() && !simd_result[i].is_nan()) {
            panic!(
                "{}: input: {:?}, actual: {} != simd: {}",
                msg, ptr[i], result[i], simd_result[i]
            );
        }
    }
}

pub(crate) fn bf16_to_f32<const N: usize, F>(ptr: [half::bf16; N], conv_func: F, msg: &str)
where
    F: Fn([half::bf16; N]) -> [f32; N],
{
    let mut result = [0.0; N];
    for i in 0..N {
        result[i] = ptr[i].to_f32();
    }
    let simd_result = conv_func(ptr);
    for i in 0..N {
        if result[i] != simd_result[i] && (!result[i].is_nan() && !simd_result[i].is_nan()) {
            panic!("{}: actual: {} != simd: {}", msg, result[i], simd_result[i]);
        }
    }
}

pub(crate) fn gen_input<T: TypeCommon + SampleUniform + std::cmp::PartialOrd>(
    rng: &mut rand::rngs::ThreadRng,
    range: core::ops::RangeInclusive<T>,
) -> T {
    use rand::Rng;
    rng.gen_range(range)
}

pub(crate) fn gen_vector_random<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd,
    const N: usize,
>(
    rng: &mut rand::rngs::ThreadRng,
    range: core::ops::RangeInclusive<T>,
) -> [T; N] {
    let mut result = [T::ZERO; N];
    for i in 0..N {
        result[i] = gen_input(rng, range.clone());
    }
    result
}

pub(crate) fn f32_to_f16_test(range: core::ops::RangeInclusive<f32>, repeats: usize, msg: &str) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<f32, { F32Vec::SIZE * 2 }>(&mut rng, range.clone());
        f32_to_f16(
            input,
            |x| {
                let mut high = F32Vec::default();
                let mut low = F32Vec::default();
                high.copy_from_slice(&x[0..F32Vec::SIZE]);
                low.copy_from_slice(&x[F32Vec::SIZE..]);
                let res = F16Vec::from_2_f32x4([high, low]);
                unsafe { std::mem::transmute(res) }
            },
            msg,
        );
    }
}

pub(crate) fn f16_to_f32_test(
    range: core::ops::RangeInclusive<half::f16>,
    repeats: usize,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<half::f16, { F16Vec::SIZE }>(&mut rng, range.clone());
        f16_to_f32(
            input,
            |x| {
                let mut val = F16Vec::default();
                val.copy_from_slice(&x);
                let res = val.to_2_f32x4();
                let mut result = [0.0; F16Vec::SIZE];
                for i in 0..F16Vec::SIZE / 2 {
                    result[i] = res[0][i];
                    result[i + F16Vec::SIZE / 2] = res[1][i];
                }
                result
            },
            msg,
        );
    }
}

pub(crate) fn f32_to_bf16_test(range: core::ops::RangeInclusive<f32>, repeats: usize, msg: &str) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<f32, { F32Vec::SIZE * 2 }>(&mut rng, range.clone());
        f32_to_bf16(
            input,
            |x| {
                let mut high = F32Vec::default();
                let mut low = F32Vec::default();
                high.copy_from_slice(&x[0..F32Vec::SIZE]);
                low.copy_from_slice(&x[F32Vec::SIZE..]);
                let res = Bf16Vec::from_2_f32vec([high, low]);
                unsafe { std::mem::transmute(res) }
            },
            msg,
        );
    }
}

pub(crate) fn bf16_to_f32_test(
    range: core::ops::RangeInclusive<half::bf16>,
    repeats: usize,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<half::bf16, { Bf16Vec::SIZE }>(&mut rng, range.clone());
        bf16_to_f32(
            input,
            |x| {
                let mut val = Bf16Vec::default();
                val.copy_from_slice(&x);
                let res = val.to_2_f32x4();
                let mut result = [0.0; Bf16Vec::SIZE];
                for i in 0..Bf16Vec::SIZE / 2 {
                    result[i] = res[0][i];
                    result[i + Bf16Vec::SIZE / 2] = res[1][i];
                }
                result
            },
            msg,
        );
    }
}

pub(crate) fn f32_to_f16_test_single(val: f32, msg: &str) {
    let res = [val; F32Vec::SIZE * 2];
    f32_to_f16::<{ F32Vec::SIZE * 2 }, _>(
        res,
        |x| {
            let mut high = F32Vec::default();
            let mut low = F32Vec::default();
            high.copy_from_slice(&x[0..F32Vec::SIZE]);
            low.copy_from_slice(&x[F32Vec::SIZE..]);
            let res = F16Vec::from_2_f32x4([high, low]);
            unsafe { std::mem::transmute(res) }
        },
        msg,
    );
}

pub(crate) fn f16_to_f32_test_single(val: half::f16, msg: &str) {
    let res = [val; F16Vec::SIZE];
    f16_to_f32::<{ F16Vec::SIZE }, _>(
        res,
        |x| {
            let mut val = F16Vec::default();
            val.copy_from_slice(&x);
            let res = val.to_2_f32x4();
            let mut result = [0.0; F16Vec::SIZE];
            for i in 0..F16Vec::SIZE / 2 {
                result[i] = res[0][i];
                result[i + F16Vec::SIZE / 2] = res[1][i];
            }
            result
        },
        msg,
    );
}

pub(crate) fn f32_to_bf16_test_single(val: f32, msg: &str) {
    let res = [val; F32Vec::SIZE * 2];
    f32_to_bf16::<{ F32Vec::SIZE * 2 }, _>(
        res,
        |x| {
            let mut high = F32Vec::default();
            let mut low = F32Vec::default();
            high.copy_from_slice(&x[0..F32Vec::SIZE]);
            low.copy_from_slice(&x[F32Vec::SIZE..]);
            let res = Bf16Vec::from_2_f32vec([high, low]);
            unsafe { std::mem::transmute(res) }
        },
        msg,
    );
}

pub(crate) fn bf16_to_f32_test_single(val: half::bf16, msg: &str) {
    let res = [val; Bf16Vec::SIZE];
    bf16_to_f32::<{ Bf16Vec::SIZE }, _>(
        res,
        |x| {
            let mut val = Bf16Vec::default();
            val.copy_from_slice(&x);
            let res = val.to_2_f32x4();
            let mut result = [0.0; Bf16Vec::SIZE];
            for i in 0..Bf16Vec::SIZE / 2 {
                result[i] = res[0][i];
                result[i + Bf16Vec::SIZE / 2] = res[1][i];
            }
            result
        },
        msg,
    );
}

pub(crate) fn test_computes_2operands<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T, T) -> T,
    op: impl Fn(T::Vec, T::Vec) -> T::Vec,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, range.clone());
        let input2 = gen_vector_random::<T, N>(&mut rng, range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let vec2 = unsafe { T::Vec::from_ptr(input2.as_ptr()) };
        let res = op(vec, vec2);
        let mut result = vec![T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i], input2[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if result[i] != simd_result[i] {
                panic!("{}: actual: {} != simd: {}", msg, result[i], simd_result[i]);
            }
        }
    }
}

pub(crate) fn test_computes_2operands_float<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    lhs_range: core::ops::RangeInclusive<T>,
    rhs_range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T, T) -> T,
    op: impl Fn(T::Vec, T::Vec) -> T::Vec,
    assert_op: impl Fn(T, T) -> bool,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, lhs_range.clone());
        let input2 = gen_vector_random::<T, N>(&mut rng, rhs_range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let vec2 = unsafe { T::Vec::from_ptr(input2.as_ptr()) };
        let res = op(vec, vec2);
        let mut result = vec![T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i], input2[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if !assert_op(result[i], simd_result[i]) {
                panic!(
                    "{}: input: {} actual: {} != simd: {}",
                    msg, input[i], result[i], simd_result[i]
                );
            }
        }
    }
}

pub(crate) fn test_computes_2operands_int<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    lhs_range: core::ops::RangeInclusive<T>,
    rhs_range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T, T) -> T,
    op: impl Fn(T::Vec, T::Vec) -> T::Vec,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, lhs_range.clone());
        let input2 = gen_vector_random::<T, N>(&mut rng, rhs_range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let vec2 = unsafe { T::Vec::from_ptr(input2.as_ptr()) };
        let res = op(vec, vec2);
        let mut result = vec![T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i], input2[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if result[i] != simd_result[i] {
                panic!(
                    "{}: input: {} actual: {} != simd: {}",
                    msg, input[i], result[i], simd_result[i]
                );
            }
        }
    }
}

pub(crate) fn test_computes_1operands_float<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T) -> T,
    op: impl Fn(T::Vec) -> T::Vec,
    assert_op: impl Fn(T, T) -> bool,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let res = op(vec);
        let mut result = [T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if !assert_op(result[i], simd_result[i]) {
                panic!(
                    "{}: input: {} actual: {} != simd: {}",
                    msg, input[i], result[i], simd_result[i]
                );
            }
        }
    }
}

pub(crate) fn test_computes_1operands_int<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T) -> T,
    op: impl Fn(T::Vec) -> T::Vec,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let res = op(vec);
        let mut result = [T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if result[i] != simd_result[i] {
                panic!(
                    "{}: input: {} actual: {} != simd: {}",
                    msg, input[i], result[i], simd_result[i]
                );
            }
        }
    }
}

pub(crate) fn test_computes_3operands<
    T: TypeCommon + SampleUniform + std::cmp::PartialOrd + std::fmt::Display,
    const N: usize,
>(
    range: core::ops::RangeInclusive<T>,
    repeats: usize,
    scalar_op: impl Fn(T, T, T) -> T,
    op: impl Fn(T::Vec, T::Vec, T::Vec) -> T::Vec,
    msg: &str,
) {
    let mut rng = rand::thread_rng();
    for _ in 0..repeats {
        let input = gen_vector_random::<T, N>(&mut rng, range.clone());
        let input2 = gen_vector_random::<T, N>(&mut rng, range.clone());
        let input3 = gen_vector_random::<T, N>(&mut rng, range.clone());
        let vec = unsafe { T::Vec::from_ptr(input.as_ptr()) };
        let vec2 = unsafe { T::Vec::from_ptr(input2.as_ptr()) };
        let vec3 = unsafe { T::Vec::from_ptr(input3.as_ptr()) };
        let res = op(vec, vec2, vec3);
        let mut result = vec![T::ZERO; N];
        for i in 0..N {
            result[i] = scalar_op(input[i], input2[i], input3[i]);
        }
        let simd_result = unsafe { std::mem::transmute::<&T::Vec, &[T; N]>(&res) };
        for i in 0..N {
            if result[i] != simd_result[i] {
                panic!("{}: actual: {} != simd: {}", msg, result[i], simd_result[i]);
            }
        }
    }
}

macro_rules! test_computes_2operands_for_type {
    ($T:ty, $size:expr, $repeats:literal, $range: expr, $msg:expr, $op:ident, $vec_op:ident) => {
        test_computes_2operands::<$T, $size>(
            $range,
            $repeats,
            |a, b| a.$op(b),
            |a, b| a.$vec_op(b),
            $msg,
        );
    };
    ($T:ty, $size:expr, $repeats:literal, $range: expr, $msg:expr, $op:ident, $vec_op:tt) => {
        test_computes_2operands::<$T, $size>(
            $range,
            $repeats,
            |a, b| a.$op(b),
            |a, b| a $vec_op b,
            $msg,
        );
    };
    ($T:ty, $size:expr, $repeats:literal, $range: expr, $msg:expr, $op:tt) => {
        test_computes_2operands::<$T, $size>(
            $range,
            $repeats,
            |a, b| a $op b,
            |a, b| a $op b,
            $msg,
        );
    };
}

macro_rules! test_computes_3operands_for_type {
    ($T:ty, $size:expr, $repeats:literal, $range: expr, $msg:expr, $op:ident, $vec_op:ident) => {
        test_computes_3operands::<$T, $size>(
            $range,
            $repeats,
            |a, b, c| a.$op(b, c),
            |a, b, c| a.$vec_op(b, c),
            $msg,
        );
    };
}

fn f16_ulp_diff(a: half::f16, b: half::f16) -> i32 {
    let a = a.to_f32();
    let b = b.to_f32();
    if a == b {
        return 0;
    }
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f32::EPSILON)).abs();
    if rel_diff < 0.005 {
        return 0;
    }
    if (a == 0.0 || a.is_subnormal()) && (b == 0.0 || b.is_subnormal()) {
        return 0;
    }
    if a == 0.0 && b != 0.0 {
        return 10000;
    }
    if (a.is_infinite() && b.is_infinite()) || (a.is_nan() && b.is_nan()) {
        return 0;
    }

    let bits_a = unsafe { std::mem::transmute::<f32, i32>(a) };
    let bits_b = unsafe { std::mem::transmute::<f32, i32>(b) };
    match bits_a.checked_sub(bits_b) {
        Some(diff) => {
            if diff == i32::MIN {
                i32::MAX
            } else {
                diff.abs()
            }
        }
        None => match bits_b.checked_sub(bits_a) {
            Some(diff) => {
                if diff == i32::MIN {
                    i32::MAX
                } else {
                    diff.abs()
                }
            }
            None => i32::MAX,
        },
    }
}

fn bf16_ulp_diff(a: half::bf16, b: half::bf16) -> i32 {
    let a = a.to_f32();
    let b = b.to_f32();
    if a == b {
        return 0;
    }
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f32::EPSILON)).abs();
    if rel_diff < 0.005 {
        return 0;
    }
    if (a == 0.0 || a.is_subnormal()) && (b == 0.0 || b.is_subnormal()) {
        return 0;
    }
    if a == 0.0 && b != 0.0 {
        return 10000;
    }
    if (a.is_infinite() && b.is_infinite()) || (a.is_nan() && b.is_nan()) {
        return 0;
    }

    let bits_a = unsafe { std::mem::transmute::<f32, i32>(a) };
    let bits_b = unsafe { std::mem::transmute::<f32, i32>(b) };
    match bits_a.checked_sub(bits_b) {
        Some(diff) => {
            if diff == i32::MIN {
                i32::MAX
            } else {
                diff.abs()
            }
        }
        None => match bits_b.checked_sub(bits_a) {
            Some(diff) => {
                if diff == i32::MIN {
                    i32::MAX
                } else {
                    diff.abs()
                }
            }
            None => i32::MAX,
        },
    }
}

fn f32_ulp_diff(a: f32, b: f32) -> i32 {
    if a == b {
        return 0;
    }
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f32::EPSILON)).abs();
    if rel_diff < 0.005 {
        return 0;
    }
    if (a == 0.0 || a.is_subnormal()) && (b == 0.0 || b.is_subnormal()) {
        return 0;
    }
    if a == 0.0 && b != 0.0 {
        return 10000;
    }
    if (a.is_infinite() && b.is_infinite()) || (a.is_nan() && b.is_nan()) {
        return 0;
    }

    let bits_a = unsafe { std::mem::transmute::<f32, i32>(a) };
    let bits_b = unsafe { std::mem::transmute::<f32, i32>(b) };
    match bits_a.checked_sub(bits_b) {
        Some(diff) => {
            if diff == i32::MIN {
                i32::MAX
            } else {
                diff.abs()
            }
        }
        None => match bits_b.checked_sub(bits_a) {
            Some(diff) => {
                if diff == i32::MIN {
                    i32::MAX
                } else {
                    diff.abs()
                }
            }
            None => i32::MAX,
        },
    }
}

fn f64_ulp_diff(a: f64, b: f64) -> i64 {
    if a == b {
        return 0;
    }
    let rel_diff = ((a - b) / (a.abs() + b.abs() + f64::EPSILON)).abs();
    if rel_diff < 0.005 {
        return 0;
    }
    if (a == 0.0 || a.is_subnormal()) && (b == 0.0 || b.is_subnormal()) {
        return 0;
    }
    if a == 0.0 && b != 0.0 {
        return 10000;
    }
    if (a.is_infinite() && b.is_infinite()) || (a.is_nan() && b.is_nan()) {
        return 0;
    }

    let bits_a = unsafe { std::mem::transmute::<f64, i64>(a) };
    let bits_b = unsafe { std::mem::transmute::<f64, i64>(b) };
    (bits_a - bits_b).abs()
}

#[cfg(test)]
mod tests {
    use crate::{
        convertion::Convertor,
        simd::sleef::common::misc::SQRT_FLT_MAX,
        traits::{SimdCompare, SimdMath},
        type_promote::{BitWiseOut, Eval2, FloatOutBinary2, FloatOutUnary2, NormalOut2, NormalOutUnary2},
    };

    use super::*;
    #[test]
    fn test_convert_f32_to_f16() {
        f32_to_f16_test(-f32::MIN..=f32::MAX, 1000, "f16::from_f32 min..max");
        f32_to_f16_test_single(f32::INFINITY, "f16::from_f32 inf");
        f32_to_f16_test_single(f32::NEG_INFINITY, "f16::from_f32 neg_inf");
        f32_to_f16_test_single(-0.0, "f16::from_f32 neg_zero");
        f32_to_f16_test_single(0.0, "f16::from_f32 zero");
        f32_to_f16_test_single(f32::MAX, "f16::from_f32 max");
        f32_to_f16_test_single(f32::MIN, "f16::from_f32 min");
        f32_to_f16_test_single(f32::NAN, "f16::from_f32 nan");
    }
    #[test]
    fn test_convert_f32_to_bf16() {
        f32_to_bf16_test(-f32::MIN..=f32::MAX, 1000, "bf16::from_f32 min..max");
        f32_to_bf16_test_single(f32::INFINITY, "bf16::from_f32 inf");
        f32_to_bf16_test_single(f32::NEG_INFINITY, "bf16::from_f32 neg_inf");
        f32_to_bf16_test_single(-0.0, "bf16::from_f32 neg_zero");
        f32_to_bf16_test_single(0.0, "bf16::from_f32 zero");
        f32_to_bf16_test_single(f32::MAX, "bf16::from_f32 max");
        f32_to_bf16_test_single(f32::MIN, "bf16::from_f32 min");
        f32_to_bf16_test_single(f32::NAN, "bf16::from_f32 nan");
    }
    #[test]
    fn test_convert_f16_to_f32() {
        f16_to_f32_test(
            -half::f16::MIN..=half::f16::MAX,
            1000,
            "f16::to_f32 min..max",
        );
        f16_to_f32_test_single(half::f16::MAX, "f16::to_f32 max");
        f16_to_f32_test_single(half::f16::MIN, "f16::to_f32 min");
        f16_to_f32_test_single(half::f16::NAN, "f16::to_f32 nan");
        f16_to_f32_test_single(half::f16::ZERO, "f16::to_f32 zero");
        f16_to_f32_test_single(half::f16::ONE, "f16::to_f32 one");
        f16_to_f32_test_single(half::f16::NEG_ZERO, "f16::to_f32 neg_zero");
        f16_to_f32_test_single(half::f16::NEG_ONE, "f16::to_f32 neg_one");
        f16_to_f32_test_single(half::f16::NEG_INFINITY, "f16::to_f32 neg_inf");
        f16_to_f32_test_single(half::f16::INFINITY, "f16::to_f32 inf");
    }
    #[test]
    fn test_convert_bf16_to_f32() {
        bf16_to_f32_test(
            -half::bf16::MIN..=half::bf16::MAX,
            1000,
            "bf16::to_f32 min..max",
        );
        bf16_to_f32_test_single(half::bf16::MAX, "bf16::to_f32 max");
        bf16_to_f32_test_single(half::bf16::MIN, "bf16::to_f32 min");
        bf16_to_f32_test_single(half::bf16::NAN, "bf16::to_f32 nan");
        bf16_to_f32_test_single(half::bf16::ZERO, "bf16::to_f32 zero");
        bf16_to_f32_test_single(half::bf16::ONE, "bf16::to_f32 one");
        bf16_to_f32_test_single(half::bf16::NEG_ZERO, "bf16::to_f32 neg_zero");
        bf16_to_f32_test_single(half::bf16::NEG_ONE, "bf16::to_f32 neg_one");
        bf16_to_f32_test_single(half::bf16::NEG_INFINITY, "bf16::to_f32 neg_inf");
        bf16_to_f32_test_single(half::bf16::INFINITY, "bf16::to_f32 inf");
    }

    #[rustfmt::skip]
    #[test]
    fn test_add() {
        test_computes_2operands_for_type!(i8, { I8Vec::SIZE }, 1000, i8::MIN..=i8::MAX, "i8::add", wrapping_add, __add);
        test_computes_2operands_for_type!(i16, { I16Vec::SIZE }, 1000, i16::MIN..=i16::MAX, "i16::add", wrapping_add, __add);
        test_computes_2operands_for_type!(i32, { I32Vec::SIZE }, 1000, i32::MIN..=i32::MAX, "i32::add", wrapping_add, __add);
        test_computes_2operands_for_type!(i64, { I64Vec::SIZE }, 1000, i64::MIN..=i64::MAX, "i64::add", wrapping_add, __add);
        test_computes_2operands_for_type!(isize, { IsizeVec::SIZE }, 1000, isize::MIN..=isize::MAX, "isize::add", wrapping_add, __add);
        test_computes_2operands_for_type!(u8, { U8Vec::SIZE }, 1000, u8::MIN..=u8::MAX, "u8::add", wrapping_add, __add);
        test_computes_2operands_for_type!(u16, { U16Vec::SIZE }, 1000, u16::MIN..=u16::MAX, "u16::add", wrapping_add, __add);
        test_computes_2operands_for_type!(u32, { U32Vec::SIZE }, 1000, u32::MIN..=u32::MAX, "u32::add", wrapping_add, __add);
        test_computes_2operands_for_type!(u64, { U64Vec::SIZE }, 1000, u64::MIN..=u64::MAX, "u64::add", wrapping_add, __add);
        test_computes_2operands_for_type!(usize, { UsizeVec::SIZE }, 1000, usize::MIN..=usize::MAX, "usize::add", wrapping_add, __add);
        use std::ops::Add;
        test_computes_2operands_for_type!(f32, { F32Vec::SIZE }, 1000, -1e15..=1e15, "f32::add", add, __add);
        test_computes_2operands_for_type!(half::f16, { F16Vec::SIZE }, 1000, half::f16::from_f32_const(i16::MIN as f32)..=half::f16::from_f32_const(i16::MAX as f32), "f16::add", add, __add);
        test_computes_2operands_for_type!(half::bf16, { Bf16Vec::SIZE }, 1000, half::bf16::from_f32_const(i16::MIN as f32)..=half::bf16::from_f32_const(i16::MAX as f32), "bf16::add", add, __add);
        test_computes_2operands_for_type!(f64, { F64Vec::SIZE }, 1000, -1e15..=1e15, "f64::add", add, __add);
    }

    #[rustfmt::skip]
    #[test]
    fn test_sub() {
        test_computes_2operands_for_type!(i8, { I8Vec::SIZE }, 1000, i8::MIN..=i8::MAX, "i8::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(i16, { I16Vec::SIZE }, 1000, i16::MIN..=i16::MAX, "i16::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(i32, { I32Vec::SIZE }, 1000, i32::MIN..=i32::MAX, "i32::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(i64, { I64Vec::SIZE }, 1000, i64::MIN..=i64::MAX, "i64::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(isize, { IsizeVec::SIZE }, 1000, isize::MIN..=isize::MAX, "isize::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(u8, { U8Vec::SIZE }, 1000, u8::MIN..=u8::MAX, "u8::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(u16, { U16Vec::SIZE }, 1000, u16::MIN..=u16::MAX, "u16::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(u32, { U32Vec::SIZE }, 1000, u32::MIN..=u32::MAX, "u32::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(u64, { U64Vec::SIZE }, 1000, u64::MIN..=u64::MAX, "u64::sub", wrapping_sub, __sub);
        test_computes_2operands_for_type!(usize, { UsizeVec::SIZE }, 1000, usize::MIN..=usize::MAX, "usize::sub", wrapping_sub, __sub);

        use std::ops::Sub;
        test_computes_2operands_for_type!(f32, { F32Vec::SIZE }, 1000, -1e15..=1e15, "f32::sub", sub, __sub);
        test_computes_2operands_for_type!(half::f16, { F16Vec::SIZE }, 1000, half::f16::from_f32_const(i16::MIN as f32)..=half::f16::from_f32_const(i16::MAX as f32), "f16::sub", sub, __sub);
        test_computes_2operands_for_type!(half::bf16, { Bf16Vec::SIZE }, 1000, half::bf16::from_f32_const(i16::MIN as f32)..=half::bf16::from_f32_const(i16::MAX as f32), "bf16::sub", sub, __sub);
        test_computes_2operands_for_type!(f64, { F64Vec::SIZE }, 1000, -1e15..=1e15, "f64::sub", sub, __sub);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mul() {
        test_computes_2operands_for_type!(i8, { I8Vec::SIZE }, 1000, 10..=10, "i8::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(i16, { I16Vec::SIZE }, 1000, 100..=100, "i16::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(i32, { I32Vec::SIZE }, 1000, 1000..=1000, "i32::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(i64, { I64Vec::SIZE }, 1000, 1000..=1000, "i64::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(isize, { IsizeVec::SIZE }, 1000, 1000..=1000, "isize::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(u8, { U8Vec::SIZE }, 1000, 10..=10, "u8::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(u16, { U16Vec::SIZE }, 1000, 100..=100, "u16::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(u32, { U32Vec::SIZE }, 1000, 1000..=1000, "u32::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(u64, { U64Vec::SIZE }, 1000, 1000..=1000, "u64::mul", wrapping_mul, __mul);
        test_computes_2operands_for_type!(usize, { UsizeVec::SIZE }, 1000, 1000..=1000, "usize::mul", wrapping_mul, __mul);

        use std::ops::Mul;
        test_computes_2operands_for_type!(f32, { F32Vec::SIZE }, 1000, -1e5..=1e5, "f32::mul", mul, __mul);
        test_computes_2operands_for_type!(half::f16, { F16Vec::SIZE }, 1000, half::f16::from_f32_const(i16::MIN as f32)..=half::f16::from_f32_const(i16::MAX as f32), "f16::mul", mul, __mul);
        test_computes_2operands_for_type!(half::bf16, { Bf16Vec::SIZE }, 1000, half::bf16::from_f32_const(i16::MIN as f32)..=half::bf16::from_f32_const(i16::MAX as f32), "bf16::mul", mul, __mul);
        test_computes_2operands_for_type!(f64, { F64Vec::SIZE }, 1000, -1e5..=1e5, "f64::mul", mul, __mul);
    }

    #[rustfmt::skip]
    #[test]
    fn test_mul_add() {
        use num_traits::real::Real;
        use num_traits::MulAdd;
        test_computes_3operands_for_type!(
            i8,
            { I8Vec::SIZE },
            1000,
            10..=10,
            "i8::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            i16,
            { I16Vec::SIZE },
            1000,
            100..=100,
            "i16::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            i32,
            { I32Vec::SIZE },
            1000,
            1000..=1000,
            "i32::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            i64,
            { I64Vec::SIZE },
            1000,
            1000..=1000,
            "i64::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            isize,
            { IsizeVec::SIZE },
            1000,
            1000..=1000,
            "isize::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            u8,
            { U8Vec::SIZE },
            1000,
            10..=10,
            "u8::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            u16,
            { U16Vec::SIZE },
            1000,
            100..=100,
            "u16::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            u32,
            { U32Vec::SIZE },
            1000,
            1000..=1000,
            "u32::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            u64,
            { U64Vec::SIZE },
            1000,
            1000..=1000,
            "u64::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            usize,
            { UsizeVec::SIZE },
            1000,
            1000..=1000,
            "usize::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            f32,
            { F32Vec::SIZE },
            1000,
            -1e5..=1e5,
            "f32::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            half::f16,
            { F16Vec::SIZE },
            1000,
            half::f16::from_f32_const(i16::MIN as f32)..=half::f16::from_f32_const(i16::MAX as f32),
            "f16::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            half::bf16,
            { Bf16Vec::SIZE },
            1000,
            half::bf16::from_f32_const(i16::MIN as f32)
                ..=half::bf16::from_f32_const(i16::MAX as f32),
            "bf16::mul_add",
            mul_add,
            __mul_add
        );
        test_computes_3operands_for_type!(
            f64,
            { F64Vec::SIZE },
            1000,
            -1e5..=1e5,
            "f64::mul_add",
            mul_add,
            __mul_add
        );
    }

    #[test]
    fn test_div() {
        test_computes_2operands_for_type!(
            i8,
            { I8Vec::SIZE },
            1000,
            1..=i8::MAX,
            "i8::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            i16,
            { I16Vec::SIZE },
            1000,
            1..=i16::MAX,
            "i16::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            i32,
            { I32Vec::SIZE },
            1000,
            1..=i32::MAX,
            "i32::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            i64,
            { I64Vec::SIZE },
            1000,
            1..=i64::MAX,
            "i64::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            isize,
            { IsizeVec::SIZE },
            1000,
            1..=isize::MAX,
            "isize::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            u8,
            { U8Vec::SIZE },
            1000,
            1..=u8::MAX,
            "u8::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            u16,
            { U16Vec::SIZE },
            1000,
            1..=u16::MAX,
            "u16::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            u32,
            { U32Vec::SIZE },
            1000,
            1..=u32::MAX,
            "u32::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            u64,
            { U64Vec::SIZE },
            1000,
            1..=u64::MAX,
            "u64::div",
            wrapping_div,
            __div
        );
        test_computes_2operands_for_type!(
            usize,
            { UsizeVec::SIZE },
            1000,
            1..=usize::MAX,
            "usize::div",
            wrapping_div,
            __div
        );
        use std::ops::Div;
        test_computes_2operands_for_type!(
            f32,
            { F32Vec::SIZE },
            1000,
            1.0..=1e15,
            "f32::div",
            div,
            __div
        );
        test_computes_2operands_for_type!(
            half::f16,
            { F16Vec::SIZE },
            1000,
            half::f16::from_f32_const(1f32)..=half::f16::from_f32_const(i16::MAX as f32),
            "f16::div",
            div,
            __div
        );
        test_computes_2operands_for_type!(
            half::bf16,
            { Bf16Vec::SIZE },
            1000,
            half::bf16::from_f32_const(1f32)..=half::bf16::from_f32_const(i16::MAX as f32),
            "bf16::div",
            div,
            __div
        );
        test_computes_2operands_for_type!(
            f64,
            { F64Vec::SIZE },
            1000,
            1.0..=1e15,
            "f64::div",
            div,
            __div
        );
    }

    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_i8_0() {
        test_computes_2operands_for_type!(i8, { I8Vec::SIZE }, 1000, 0..=0, "i8::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_i16_0() {
        test_computes_2operands_for_type!(i16, { I16Vec::SIZE }, 1000, 0..=0, "i16::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_i32_0() {
        test_computes_2operands_for_type!(i32, { I32Vec::SIZE }, 1000, 0..=0, "i32::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_i64_0() {
        test_computes_2operands_for_type!(i64, { I64Vec::SIZE }, 1000, 0..=0, "i64::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_isize_0() {
        test_computes_2operands_for_type!(isize, { IsizeVec::SIZE }, 1000, 0..=0, "isize::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_u8_0() {
        test_computes_2operands_for_type!(u8, { U8Vec::SIZE }, 1000, 0..=0, "u8::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_u16_0() {
        test_computes_2operands_for_type!(u16, { U16Vec::SIZE }, 1000, 0..=0, "u16::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_u32_0() {
        test_computes_2operands_for_type!(u32, { U32Vec::SIZE }, 1000, 0..=0, "u32::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_u64_0() {
        test_computes_2operands_for_type!(u64, { U64Vec::SIZE }, 1000, 0..=0, "u64::div", wrapping_div, /);
    }
    #[should_panic(expected = "division by zero")]
    #[test]
    fn test_div_usize_0() {
        test_computes_2operands_for_type!(usize, { UsizeVec::SIZE }, 1000, 0..=0, "usize::div", wrapping_div, /);
    }
    #[test]
    fn test_div_float_0() {
        macro_rules! test_div_float_0 {
            ($type_vec:ident, $type:ty, $msg:expr) => {
                let res = $type_vec::splat(<$type>::ZERO) / $type_vec::splat(<$type>::ZERO);
                let simd_result =
                    unsafe { std::mem::transmute::<&$type_vec, &[$type; $type_vec::SIZE]>(&res) };
                for i in 0..$type_vec::SIZE {
                    assert!(simd_result[i].is_nan(), $msg);
                }
            };
        }
        test_div_float_0!(F32Vec, f32, "f32::div: simd_result[i] is not nan");
        test_div_float_0!(F64Vec, f64, "f64::div: simd_result[i] is not nan");
        test_div_float_0!(F16Vec, half::f16, "f16::div: simd_result[i] is not nan");
        test_div_float_0!(Bf16Vec, half::bf16, "bf16::div: simd_result[i] is not nan");
    }
    #[test]
    fn test_div_float_nan() {
        macro_rules! test_div_float_nan {
            ($type_vec:ident, $type:ty, $msg:expr) => {
                let res = $type_vec::splat(<$type>::NAN) / $type_vec::splat(<$type>::NAN);
                let simd_result =
                    unsafe { std::mem::transmute::<&$type_vec, &[$type; $type_vec::SIZE]>(&res) };
                for i in 0..$type_vec::SIZE {
                    assert!(simd_result[i].is_nan(), $msg);
                }
                let res = $type_vec::splat(<$type>::ZERO) / $type_vec::splat(<$type>::NAN);
                let simd_result =
                    unsafe { std::mem::transmute::<&$type_vec, &[$type; $type_vec::SIZE]>(&res) };
                for i in 0..$type_vec::SIZE {
                    assert!(simd_result[i].is_nan(), $msg);
                }
                let res = $type_vec::splat(<$type>::NAN) / $type_vec::splat(<$type>::ONE);
                let simd_result =
                    unsafe { std::mem::transmute::<&$type_vec, &[$type; $type_vec::SIZE]>(&res) };
                for i in 0..$type_vec::SIZE {
                    assert!(simd_result[i].is_nan(), $msg);
                }
            };
        }
        test_div_float_nan!(F32Vec, f32, "f32::div: simd_result[i] is not nan");
        test_div_float_nan!(F64Vec, f64, "f64::div: simd_result[i] is not nan");
        test_div_float_nan!(F16Vec, half::f16, "f16::div: simd_result[i] is not nan");
        test_div_float_nan!(Bf16Vec, half::bf16, "bf16::div: simd_result[i] is not nan");
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_eq() {
        macro_rules! test_simd_eq {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 == 0: $val:literal, $val2: literal == 0: $val3:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_eq(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_eq($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 == 0: $val:literal, $val2: literal == 0: $val3:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_eq(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_eq($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
        }
        test_simd_eq!(I8Vec, I8Vec, i8, "i8::simd_eq: simd_result[i] is not -1", "i8::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(I16Vec, I16Vec, i16, "i16::simd_eq: simd_result[i] is not -1", "i16::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(I32Vec, I32Vec, i32, "i32::simd_eq: simd_result[i] is not -1", "i32::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(I64Vec, I64Vec, i64, "i64::simd_eq: simd_result[i] is not -1", "i64::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(IsizeVec, IsizeVec, isize, "isize::simd_eq: simd_result[i] is not -1", "isize::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(U8Vec, I8Vec, u8, "u8::simd_eq: simd_result[i] is not -1", "u8::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(U16Vec, I16Vec, u16, "u16::simd_eq: simd_result[i] is not -1", "u16::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(U32Vec, I32Vec, u32, "u32::simd_eq: simd_result[i] is not -1", "u32::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(U64Vec, I64Vec, u64, "u64::simd_eq: simd_result[i] is not -1", "u64::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(UsizeVec, IsizeVec, usize, "usize::simd_eq: simd_result[i] is not -1", "usize::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10 == 0: 0);
        test_simd_eq!(F32Vec, I32Vec, f32, "f32::simd_eq: simd_result[i] is not -1", "f32::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10.0 == 0: 0);
        test_simd_eq!(F64Vec, I64Vec, f64, "f64::simd_eq: simd_result[i] is not -1", "f64::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10.0 == 0: 0);
        test_simd_eq!(2, F16Vec, I16Vec, half::f16, "f16::simd_eq: simd_result[i] is not -1", "f16::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10.0 == 0: 0);
        test_simd_eq!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_eq: simd_result[i] is not -1", "bf16::simd_eq: simd_result[i] is not 0", 0 == 0: -1, 10.0 == 0: 0);
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_ne() {
        macro_rules! test_simd_ne {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 != 0: $val:literal, $val2: literal != 0: $val3:expr) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_ne(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_ne($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 != 0: $val:literal, $val2: literal != 0: $val3:expr) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_ne(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_ne($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
        }
        test_simd_ne!(I8Vec, I8Vec, i8, "i8::simd_ne: simd_result[i] is not 0", "i8::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(I16Vec, I16Vec, i16, "i16::simd_ne: simd_result[i] is not 0", "i16::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(I32Vec, I32Vec, i32, "i32::simd_ne: simd_result[i] is not 0", "i32::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(I64Vec, I64Vec, i64, "i64::simd_ne: simd_result[i] is not 0", "i64::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(IsizeVec, IsizeVec, isize, "isize::simd_ne: simd_result[i] is not 0", "isize::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(U8Vec, I8Vec, u8, "u8::simd_ne: simd_result[i] is not 0", "u8::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(U16Vec, I16Vec, u16, "u16::simd_ne: simd_result[i] is not 0", "u16::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(U32Vec, I32Vec, u32, "u32::simd_ne: simd_result[i] is not 0", "u32::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(U64Vec, I64Vec, u64, "u64::simd_ne: simd_result[i] is not 0", "u64::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(UsizeVec, IsizeVec, usize, "usize::simd_ne: simd_result[i] is not 0", "usize::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10 != 0: -1);
        test_simd_ne!(F32Vec, I32Vec, f32, "f32::simd_ne: simd_result[i] is not 0", "f32::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10.0 != 0: -1);
        test_simd_ne!(F64Vec, I64Vec, f64, "f64::simd_ne: simd_result[i] is not 0", "f64::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10.0 != 0: -1);
        test_simd_ne!(2, F16Vec, I16Vec, half::f16, "f16::simd_ne: simd_result[i] is not 0", "f16::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10.0 != 0: -1);
        test_simd_ne!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_ne: simd_result[i] is not 0", "bf16::simd_ne: simd_result[i] is not -1", 0 != 0: 0, 10.0 != 0: -1);
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_gt() {
        macro_rules! test_simd_gt {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 > 0: $val:literal, $val2: literal > 0: $val3:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_gt(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_gt($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 > 0: $val:literal, $val2: literal > 0: $val3:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_gt(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_gt($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);
            };
        }
        test_simd_gt!(I8Vec, I8Vec, i8, "i8::simd_gt: simd_result[i] is not 0", "i8::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(I16Vec, I16Vec, i16, "i16::simd_gt: simd_result[i] is not 0", "i16::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(I32Vec, I32Vec, i32, "i32::simd_gt: simd_result[i] is not 0", "i32::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(I64Vec, I64Vec, i64, "i64::simd_gt: simd_result[i] is not 0", "i64::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(IsizeVec, IsizeVec, isize, "isize::simd_gt: simd_result[i] is not 0", "isize::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(U8Vec, I8Vec, u8, "u8::simd_gt: simd_result[i] is not 0", "u8::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(U16Vec, I16Vec, u16, "u16::simd_gt: simd_result[i] is not 0", "u16::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(U32Vec, I32Vec, u32, "u32::simd_gt: simd_result[i] is not 0", "u32::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(U64Vec, I64Vec, u64, "u64::simd_gt: simd_result[i] is not 0", "u64::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(UsizeVec, IsizeVec, usize, "usize::simd_gt: simd_result[i] is not 0", "usize::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10 > 0: -1);
        test_simd_gt!(F32Vec, I32Vec, f32, "f32::simd_gt: simd_result[i] is not 0", "f32::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10.0 > 0: -1);
        test_simd_gt!(F64Vec, I64Vec, f64, "f64::simd_gt: simd_result[i] is not 0", "f64::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10.0 > 0: -1);
        test_simd_gt!(2, F16Vec, I16Vec, half::f16, "f16::simd_gt: simd_result[i] is not 0", "f16::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10.0 > 0: -1);
        test_simd_gt!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_gt: simd_result[i] is not 0", "bf16::simd_gt: simd_result[i] is not -1", 0 > 0: 0, 10.0 > 0: -1);
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_ge() {
        macro_rules! test_simd_ge {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, $msg3:expr, 0 >= 0: $val:literal, $val2: literal >= 0: $val3:expr, $val4:literal >= $val5:literal: $val6:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_ge(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_ge($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);

                let input = [$val4; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let input2 = [$val5; $type_vec::SIZE];
                let mut input_vec2 = $type_vec::default();
                input_vec2.copy_from_slice(&input2);
                let res = input_vec.simd_ge(input_vec2);
                assert_eq!(res, $mask_type::splat($val6), $msg3);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, $msg3:expr, 0 >= 0: $val:literal, $val2: literal >= 0: $val3:expr, $val4:literal >= $val5:literal: $val6:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_ge(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_ge($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);

                let input = [<$type>::from_f32_const($val4); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let input2 = [<$type>::from_f32_const($val5); $type_vec::SIZE];
                let mut input_vec2 = $type_vec::default();
                input_vec2.copy_from_slice(&input2);
                let res = input_vec.simd_ge(input_vec2);
                assert_eq!(res, $mask_type::splat($val6), $msg3);
            };
        }
        test_simd_ge!(I8Vec, I8Vec, i8, "i8::simd_ge: simd_result[i] is not 0", "i8::simd_ge: simd_result[i] is not -1", "i8::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, -1 >= 0: 0);
        test_simd_ge!(I16Vec, I16Vec, i16, "i16::simd_ge: simd_result[i] is not 0", "i16::simd_ge: simd_result[i] is not -1", "i16::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, -1 >= 0: 0);
        test_simd_ge!(I32Vec, I32Vec, i32, "i32::simd_ge: simd_result[i] is not 0", "i32::simd_ge: simd_result[i] is not -1", "i32::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, -1 >= 0: 0);
        test_simd_ge!(I64Vec, I64Vec, i64, "i64::simd_ge: simd_result[i] is not 0", "i64::simd_ge: simd_result[i] is not -1", "i64::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, -1 >= 0: 0);
        test_simd_ge!(IsizeVec, IsizeVec, isize, "isize::simd_ge: simd_result[i] is not 0", "isize::simd_ge: simd_result[i] is not -1", "isize::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, -1 >= 0: 0);
        test_simd_ge!(U8Vec, I8Vec, u8, "u8::simd_ge: simd_result[i] is not 0", "u8::simd_ge: simd_result[i] is not -1", "u8::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, 0 >= 2: 0);
        test_simd_ge!(U16Vec, I16Vec, u16, "u16::simd_ge: simd_result[i] is not 0", "u16::simd_ge: simd_result[i] is not -1", "u16::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, 0 >= 2: 0);
        test_simd_ge!(U32Vec, I32Vec, u32, "u32::simd_ge: simd_result[i] is not 0", "u32::simd_ge: simd_result[i] is not -1", "u32::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, 0 >= 2: 0); 
        test_simd_ge!(U64Vec, I64Vec, u64, "u64::simd_ge: simd_result[i] is not 0", "u64::simd_ge: simd_result[i] is not -1", "u64::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, 0 >= 2: 0);
        test_simd_ge!(UsizeVec, IsizeVec, usize, "usize::simd_ge: simd_result[i] is not 0", "usize::simd_ge: simd_result[i] is not -1", "usize::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10 >= 0: -1, 0 >= 2: 0);
        test_simd_ge!(F32Vec, I32Vec, f32, "f32::simd_ge: simd_result[i] is not 0", "f32::simd_ge: simd_result[i] is not -1", "f32::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10.0 >= 0: -1, -1.0 >= 0.0: 0);
        test_simd_ge!(F64Vec, I64Vec, f64, "f64::simd_ge: simd_result[i] is not 0", "f64::simd_ge: simd_result[i] is not -1", "f64::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10.0 >= 0: -1, -1.0 >= 0.0: 0);
        test_simd_ge!(2, F16Vec, I16Vec, half::f16, "f16::simd_ge: simd_result[i] is not 0", "f16::simd_ge: simd_result[i] is not -1", "f16::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10.0 >= 0: -1, -1.0 >= 0.0: 0);
        test_simd_ge!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_ge: simd_result[i] is not 0", "bf16::simd_ge: simd_result[i] is not -1", "bf16::simd_ge: simd_result[i] is not 1", 0 >= 0: -1, 10.0 >= 0: -1, -1.0 >= 0.0: 0);
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_lt() {
        macro_rules! test_simd_lt {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 < 0: $val:literal, $val2: literal < $val3:literal: $val4:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_lt(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_lt(unsafe { $type_vec::from_ptr([$val3; $type_vec::SIZE].as_ptr()) });
                assert_eq!(res, $mask_type::splat($val4), $msg2);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, 0 < 0: $val:literal, $val2: literal < $val3:literal: $val4:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_lt(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_lt(unsafe { $type_vec::from_ptr([<$type>::from_f32_const($val3); $type_vec::SIZE].as_ptr()) });
                assert_eq!(res, $mask_type::splat($val4), $msg2);
            };
        }
        test_simd_lt!(I8Vec, I8Vec, i8, "i8::simd_lt: simd_result[i] is not 0", "i8::simd_lt: simd_result[i] is not -1", 0 < 0: 0, -10 < 0: -1);
        test_simd_lt!(I16Vec, I16Vec, i16, "i16::simd_lt: simd_result[i] is not 0", "i16::simd_lt: simd_result[i] is not -1", 0 < 0: 0, -10 < 0: -1);
        test_simd_lt!(I32Vec, I32Vec, i32, "i32::simd_lt: simd_result[i] is not 0", "i32::simd_lt: simd_result[i] is not -1", 0 < 0: 0, -10 < 0: -1);
        test_simd_lt!(I64Vec, I64Vec, i64, "i64::simd_lt: simd_result[i] is not 0", "i64::simd_lt: simd_result[i] is not -1", 0 < 0: 0, -10 < 0: -1);
        test_simd_lt!(IsizeVec, IsizeVec, isize, "isize::simd_lt: simd_result[i] is not 0", "isize::simd_lt: simd_result[i] is not -1", 0 < 0: 0, -10 < 0: -1);
        test_simd_lt!(U8Vec, I8Vec, u8, "u8::simd_lt: simd_result[i] is not 0", "u8::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10 < 100: -1);
        test_simd_lt!(U16Vec, I16Vec, u16, "u16::simd_lt: simd_result[i] is not 0", "u16::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10 < 100: -1);
        test_simd_lt!(U32Vec, I32Vec, u32, "u32::simd_lt: simd_result[i] is not 0", "u32::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10 < 100: -1);
        test_simd_lt!(U64Vec, I64Vec, u64, "u64::simd_lt: simd_result[i] is not 0", "u64::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10 < 100: -1);
        test_simd_lt!(UsizeVec, IsizeVec, usize, "usize::simd_lt: simd_result[i] is not 0", "usize::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10 < 100: -1);
        test_simd_lt!(F32Vec, I32Vec, f32, "f32::simd_lt: simd_result[i] is not 0", "f32::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10.0 < 100.0: -1);
        test_simd_lt!(F64Vec, I64Vec, f64, "f64::simd_lt: simd_result[i] is not 0", "f64::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10.0 < 100.0: -1);
        test_simd_lt!(2, F16Vec, I16Vec, half::f16, "f16::simd_lt: simd_result[i] is not 0", "f16::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10.0 < 100.0: -1);
        test_simd_lt!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_lt: simd_result[i] is not 0", "bf16::simd_lt: simd_result[i] is not -1", 0 < 0: 0, 10.0 < 100.0: -1);
    }

    #[rustfmt::skip]
    #[test]
    fn test_simd_le() {
        macro_rules! test_simd_le {
            ($type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, $msg3:expr, 0 <= 0: $val:literal, $val2: literal <= 0: $val3:expr, $val4:literal <= $val5:literal: $val6:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_le(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [$val2; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_le($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);

                let input = [$val4; $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let input2 = [$val5; $type_vec::SIZE];
                let mut input_vec2 = $type_vec::default();
                input_vec2.copy_from_slice(&input2);
                let res = input_vec.simd_le(input_vec2);
                assert_eq!(res, $mask_type::splat($val6), $msg3);
            };
            (2, $type_vec:ident, $mask_type:ident, $type:ty, $msg1:expr, $msg2:expr, $msg3:expr, 0 <= 0: $val:literal, $val2: literal <= 0: $val3:expr, $val4:literal <= $val5:literal: $val6:literal) => {
                let mut input_vec = $type_vec::default();
                let res = input_vec.simd_le(input_vec);
                assert_eq!(res, $mask_type::splat($val), $msg1);
        
                let input = [<$type>::from_f32_const($val2); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let res = input_vec.simd_le($type_vec::default());
                assert_eq!(res, $mask_type::splat($val3), $msg2);

                let input = [<$type>::from_f32_const($val4); $type_vec::SIZE];
                input_vec.copy_from_slice(&input);
                let input2 = [<$type>::from_f32_const($val5); $type_vec::SIZE];
                let mut input_vec2 = $type_vec::default();
                input_vec2.copy_from_slice(&input2);
                let res = input_vec.simd_le(input_vec2);
                assert_eq!(res, $mask_type::splat($val6), $msg3);
            };
        }
        test_simd_le!(I8Vec, I8Vec, i8, "i8::simd_le: simd_result[i] is not 0", "i8::simd_le: simd_result[i] is not -1", "i8::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, -1 <= 0: -1);
        test_simd_le!(I16Vec, I16Vec, i16, "i16::simd_le: simd_result[i] is not 0", "i16::simd_le: simd_result[i] is not -1", "i16::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, -1 <= 0: -1);
        test_simd_le!(I32Vec, I32Vec, i32, "i32::simd_le: simd_result[i] is not 0", "i32::simd_le: simd_result[i] is not -1", "i32::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, -1 <= 0: -1);
        test_simd_le!(I64Vec, I64Vec, i64, "i64::simd_le: simd_result[i] is not -1", "i64::simd_le: simd_result[i] is not -1", "i64::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, -1 <= 0: -1);
        test_simd_le!(IsizeVec, IsizeVec, isize, "isize::simd_le: simd_result[i] is not 0", "isize::simd_le: simd_result[i] is not -1", "isize::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, -1 <= 0: -1);
        test_simd_le!(U8Vec, I8Vec, u8, "u8::simd_le: simd_result[i] is not 0", "u8::simd_le: simd_result[i] is not -1", "u8::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, 0 <= 2: -1);
        test_simd_le!(U16Vec, I16Vec, u16, "u16::simd_le: simd_result[i] is not 0", "u16::simd_le: simd_result[i] is not -1", "u16::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, 0 <= 2: -1);
        test_simd_le!(U32Vec, I32Vec, u32, "u32::simd_le: simd_result[i] is not 0", "u32::simd_le: simd_result[i] is not -1", "u32::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, 0 <= 2: -1); 
        test_simd_le!(U64Vec, I64Vec, u64, "u64::simd_le: simd_result[i] is not 0", "u64::simd_le: simd_result[i] is not -1", "u64::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, 0 <= 2: -1);
        test_simd_le!(UsizeVec, IsizeVec, usize, "usize::simd_le: simd_result[i] is not 0", "usize::simd_le: simd_result[i] is not -1", "usize::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10 <= 0: 0, 0 <= 2: -1);
        test_simd_le!(F32Vec, I32Vec, f32, "f32::simd_le: simd_result[i] is not 0", "f32::simd_le: simd_result[i] is not -1", "f32::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10.0 <= 0: 0, -1.0 <= 0.0: -1);
        test_simd_le!(F64Vec, I64Vec, f64, "f64::simd_le: simd_result[i] is not 0", "f64::simd_le: simd_result[i] is not -1", "f64::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10.0 <= 0: 0, -1.0 <= 0.0: -1);
        test_simd_le!(2, F16Vec, I16Vec, half::f16, "f16::simd_le: simd_result[i] is not 0", "f16::simd_le: simd_result[i] is not -1", "f16::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10.0 <= 0: 0, -1.0 <= 0.0: -1);
        test_simd_le!(2, Bf16Vec, I16Vec, half::bf16, "bf16::simd_le: simd_result[i] is not 0", "bf16::simd_le: simd_result[i] is not -1", "bf16::simd_le: simd_result[i] is not 1", 0 <= 0: -1, 10.0 <= 0: 0, -1.0 <= 0.0: -1);
    }

    #[rustfmt::skip]
    #[test]
    fn test_float_simd_math() {
        macro_rules! test_float_simd_math {
            ($type:ty, $size:expr, $range:expr, $repeat:expr, $tolerance: literal, $op:ident) => {
                test_computes_1operands_float::<$type, { $size }>(
                    $range,
                    $repeat,
                    |x| x.$op(),
                    |x| SimdMath::$op(x),
                    |a, b| {
                        paste::paste! {
                            [<$type _ulp_diff>](a, b) <= $tolerance
                        }
                    },
                    stringify!($type::$op),
                );
            };
            ($type:ty, $size:expr, $range:expr, $repeat:expr, $tolerance: literal, $scalar_op:expr, $vec_op:expr) => {
                test_computes_1operands_float::<$type, { $size }>(
                    $range,
                    $repeat,
                    $scalar_op,
                    $vec_op,
                    |a, b| {
                        paste::paste! {
                            [<$type _ulp_diff>](a, b) <= $tolerance
                        }
                    },
                    stringify!($type::$scalar_op),
                );
            };
        }
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, sin);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, cos);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, tan);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, asin);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, acos);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, atan);
        test_float_simd_math!(f32, F32Vec::SIZE, -88.5..=88.5, 1000, 1, sinh);
        test_float_simd_math!(f32, F32Vec::SIZE, -88.5..=88.5, 1000, 1, cosh);
        test_float_simd_math!(f32, F32Vec::SIZE, -8.7..=8.7, 1000, 1, tanh);
        test_float_simd_math!(f32, F32Vec::SIZE, -SQRT_FLT_MAX as f32..=SQRT_FLT_MAX as f32, 1000, 1, asinh);
        test_float_simd_math!(f32, F32Vec::SIZE, -SQRT_FLT_MAX as f32..=SQRT_FLT_MAX as f32, 1000, 1, acosh);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, atanh);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e19..=1e19, 1000, 1, |x| x * x, |x| SimdMath::square(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e19..=1e19, 1000, 1, |x| x * x, |x| x.__square());
        test_float_simd_math!(f32, F32Vec::SIZE, 0.0..=1e37, 1000, 1, sqrt);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, abs);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.abs(), |x| x.__abs());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, floor);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.floor(), |x| x.__floor());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, ceil);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.ceil(), |x| x.__ceil());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| -x, |x| SimdMath::neg(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| -x, |x| x.__neg());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, round);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.round(), |x| x.__round());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, signum);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.signum(), |x| x.__signum());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| if x > 0.0 {x} else {0.0}, |x| SimdMath::relu(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| if x > 0.0 {x} else {0.0}, |x| x.__relu());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.max(0.0).min(6.0), |x| SimdMath::relu6(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.max(0.0).min(6.0), |x| x.__relu6());
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, exp);
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x.exp2(), |x| SimdMath::exp2(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x.exp_m1(), |x| x.expm1());
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, ln);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| x.ln(), |x| SimdMath::log(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, log2);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, log10);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, cbrt);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, |x| libm::erff(x), |x| SimdMath::erf(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, trunc);
        test_float_simd_math!(f32, F32Vec::SIZE, -1e37..=1e37, 1000, 1, recip);
        test_float_simd_math!(f32, F32Vec::SIZE, -0.9999999..=1e37, 1000, 1, |x| x.ln_1p(), |x| SimdMath::log1p(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| 1.0 / (1.0 + (-x).exp()), |x| SimdMath::sigmoid(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -10.0..=10.0, 1000, 1, |x| (0.2 * x + 0.5).min(1.0).max(0.0), |x| SimdMath::hard_sigmoid(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, 
            |x| 0.5 * x * (libm::erff(x * std::f32::consts::FRAC_1_SQRT_2) + 1.0),
            |x| SimdMath::gelu(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -4.0..=4.0, 1000, 1, |x| x * (x + 3.0).__relu6() * (1.0 / 6.0), |x| SimdMath::hard_swish(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| (1.0 + x.exp()).ln(), |x| SimdMath::softplus(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x / (1.0 + x.abs()), |x| SimdMath::softsign(x));
        test_float_simd_math!(f32, F32Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x * x.__softplus().tanh(), |x| SimdMath::mish(x));

        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, sin);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, cos);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, tan);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, asin);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, acos);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, atan);
        test_float_simd_math!(f64, F64Vec::SIZE, -709.0..=709.0, 1000, 1, sinh);
        test_float_simd_math!(f64, F64Vec::SIZE, -709.0..=709.0, 1000, 1, cosh);
        test_float_simd_math!(f64, F64Vec::SIZE, -19.0..=19.0, 1000, 1, tanh);
        test_float_simd_math!(f64, F64Vec::SIZE, -SQRT_FLT_MAX as f64..=SQRT_FLT_MAX as f64, 1000, 1, asinh);
        test_float_simd_math!(f64, F64Vec::SIZE, -SQRT_FLT_MAX as f64..=SQRT_FLT_MAX as f64, 1000, 1, acosh);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, atanh);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e150..=1e150, 1000, 1, |x| x * x, |x| SimdMath::square(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e150..=1e150, 1000, 1, |x| x * x, |x| x.__square());
        test_float_simd_math!(f64, F64Vec::SIZE, 0.0..=1e300, 1000, 1, sqrt);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, abs);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.abs(), |x| x.__abs());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, floor);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.floor(), |x| x.__floor());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, ceil);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.ceil(), |x| x.__ceil());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| -x, |x| SimdMath::neg(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| -x, |x| x.__neg());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, round);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.round(), |x| x.__round());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, signum);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.signum(), |x| x.__signum());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| if x > 0.0 {x} else {0.0}, |x| SimdMath::relu(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| if x > 0.0 {x} else {0.0}, |x| x.__relu());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.max(0.0).min(6.0), |x| SimdMath::relu6(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.max(0.0).min(6.0), |x| x.__relu6());
        test_float_simd_math!(f64, F64Vec::SIZE, -708.0..=708.0, 1000, 1, exp);
        test_float_simd_math!(f64, F64Vec::SIZE, -708.0..=708.0, 1000, 1, |x| x.exp2(), |x| SimdMath::exp2(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -708.0..=708.0, 1000, 1, |x| x.exp_m1(), |x| x.expm1());
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, ln);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| x.ln(), |x| SimdMath::log(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, log2);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, log10);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, cbrt);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, |x| libm::erf(x), |x| SimdMath::erf(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, trunc);
        test_float_simd_math!(f64, F64Vec::SIZE, -1e306..=1e306, 1000, 1, recip);
        test_float_simd_math!(f64, F64Vec::SIZE, -0.9999999999999999..=1e306, 1000, 1, |x| x.ln_1p(), |x| SimdMath::log1p(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -88.0..=88.0, 1000, 1, |x| 1.0 / (1.0 + (-x).exp()), |x| SimdMath::sigmoid(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -10.0..=10.0, 1000, 1, |x| (0.2 * x + 0.5).min(1.0).max(0.0), |x| SimdMath::hard_sigmoid(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -88.0..=88.0, 1000, 1, 
            |x| 0.5 * x * (libm::erf(x * std::f64::consts::FRAC_1_SQRT_2) + 1.0),
            |x| SimdMath::gelu(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -4.0..=4.0, 1000, 1, |x| x * (x + 3.0).__relu6() * (1.0 / 6.0), |x| SimdMath::hard_swish(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -88.0..=88.0, 1000, 1, |x| (1.0 + x.exp()).ln(), |x| SimdMath::softplus(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x / (1.0 + x.abs()), |x| SimdMath::softsign(x));
        test_float_simd_math!(f64, F64Vec::SIZE, -88.0..=88.0, 1000, 1, |x| x * x.__softplus().tanh(), |x| SimdMath::mish(x));

        use num_traits::Float;
        use half::bf16;
        let range1 = bf16::from_f32(-1e4)..=bf16::from_f32(1e4);
        let range2 = bf16::from_f32(-11.0)..=bf16::from_f32(11.0);
        let range3 = bf16::from_f32(-8.7)..=bf16::from_f32(8.7);
        let range4 = bf16::from_f32(-130.0)..=bf16::from_f32(130.0);
        let range5 = bf16::from_f32(0.0)..=bf16::from_f32(6e4);
        let range7 = bf16::from_f32(-0.999)..=bf16::from_f32(6e4);
        let hard_sigmoid_range = bf16::from_f32(-5.0)..=bf16::from_f32(5.0);
        let hard_swish_range = bf16::from_f32(-4.0)..=bf16::from_f32(4.0);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, sin);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, cos);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, tan);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, asin);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, acos);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, atan);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, sinh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, cosh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range3.clone(), 1000, 1, tanh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range4.clone(), 1000, 1, asinh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range4.clone(), 1000, 1, acosh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, atanh);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range4.clone(), 1000, 1, |x| x * x, |x| SimdMath::square(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range4.clone(), 1000, 1, |x| x * x, |x| x.__square());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range5.clone(), 1000, 1, sqrt);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, abs);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.abs(), |x| x.__abs());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, floor);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.floor(), |x| x.__floor());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, ceil);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.ceil(), |x| x.__ceil());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| -x, |x| SimdMath::neg(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| -x, |x| x.__neg());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, round);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.round(), |x| x.__round());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, signum);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.signum(), |x| x.__signum());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| if x > half::bf16::ZERO {x} else {half::bf16::ZERO}, |x| SimdMath::relu(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.max(half::bf16::ZERO).min(half::bf16::from_f32(6.0)), |x| SimdMath::relu6(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, exp);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| x.exp2(), |x| SimdMath::exp2(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| x.exp_m1(), |x| x.expm1());
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, ln);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| x.ln(), |x| SimdMath::log(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, log2);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, log10);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, cbrt);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, |x| libm::erff(x.to_f32()).to_bf16(), |x| SimdMath::erf(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, trunc);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range1.clone(), 1000, 1, recip);
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range7.clone(), 1000, 1, |x| x.ln_1p(), |x| SimdMath::log1p(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| (1.0 / (1.0 + (-x.to_f32()).exp())).to_bf16(), |x| SimdMath::sigmoid(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, hard_sigmoid_range.clone(), 10, 1, |x| (0.2 * x.to_f32() + 0.5).min(1.0).max(0.0).to_bf16(), |x| SimdMath::hard_sigmoid(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1,
            |x| (0.5 * x.to_f32() * (libm::erff(x.to_f32() * std::f32::consts::FRAC_1_SQRT_2) + 1.0)).to_bf16(),
            |x| SimdMath::gelu(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, hard_swish_range.clone(), 1000, 1, |x| (x.to_f32() * (x.to_f32() + 3.0).__relu6() * (1.0 / 6.0)).to_bf16(), |x| SimdMath::hard_swish(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| (1.0 + x.to_f32().exp()).ln().to_bf16(), |x| SimdMath::softplus(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| (x.to_f32() / (1.0 + x.to_f32().abs())).to_bf16(), |x| SimdMath::softsign(x));
        test_float_simd_math!(bf16, Bf16Vec::SIZE, range2.clone(), 1000, 1, |x| (x.to_f32() * x.to_f32().__softplus().tanh()).to_bf16(), |x| SimdMath::mish(x));

        use half::f16;
        let range1 = half::f16::from_f32(-1e4)..=half::f16::from_f32(1e4);
        let range2 = half::f16::from_f32(-11.0)..=half::f16::from_f32(11.0);
        let range3 = half::f16::from_f32(-8.7)..=half::f16::from_f32(8.7);
        let range4 = half::f16::from_f32(-130.0)..=half::f16::from_f32(130.0);
        let range5 = half::f16::from_f32(0.0)..=half::f16::from_f32(6e4);
        let range7 = half::f16::from_f32(-0.999)..=half::f16::from_f32(6e4);
        let hard_sigmoid_range = half::f16::from_f32(-5.0)..=half::f16::from_f32(5.0);
        let hard_swish_range = half::f16::from_f32(-4.0)..=half::f16::from_f32(4.0);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, sin);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, cos);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, tan);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, asin);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, acos);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, atan);
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, sinh);
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, cosh);
        test_float_simd_math!(f16, F16Vec::SIZE, range3.clone(), 1000, 1, tanh);
        test_float_simd_math!(f16, F16Vec::SIZE, range4.clone(), 1000, 1, asinh);
        test_float_simd_math!(f16, F16Vec::SIZE, range4.clone(), 1000, 1, acosh);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, atanh);
        test_float_simd_math!(f16, F16Vec::SIZE, range4.clone(), 1000, 1, |x| x * x, |x| SimdMath::square(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range4.clone(), 1000, 1, |x| x * x, |x| x.__square());
        test_float_simd_math!(f16, F16Vec::SIZE, range5.clone(), 1000, 1, sqrt);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, abs);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.abs(), |x| x.__abs());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, floor);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.floor(), |x| x.__floor());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, ceil);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.ceil(), |x| x.__ceil());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| -x, |x| SimdMath::neg(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| -x, |x| x.__neg());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, round);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.round(), |x| x.__round());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, signum);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.signum(), |x| x.__signum());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| if x > half::f16::ZERO {x} else {half::f16::ZERO}, |x| SimdMath::relu(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.max(half::f16::ZERO).min(half::f16::from_f32(6.0)), |x| SimdMath::relu6(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, exp);
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| x.exp2(), |x| SimdMath::exp2(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| x.exp_m1(), |x| x.expm1());
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, ln);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| x.ln(), |x| SimdMath::log(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, log2);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, log10);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, cbrt);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, |x| libm::erff(x.to_f32()).to_f16(), |x| SimdMath::erf(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, trunc);
        test_float_simd_math!(f16, F16Vec::SIZE, range1.clone(), 1000, 1, recip);
        test_float_simd_math!(f16, F16Vec::SIZE, range7.clone(), 1000, 1, |x| x.ln_1p(), |x| SimdMath::log1p(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| (1.0 / (1.0 + (-x.to_f32()).exp())).to_f16(), |x| SimdMath::sigmoid(x));
        test_float_simd_math!(f16, F16Vec::SIZE, hard_sigmoid_range.clone(), 10, 1, |x| (0.2 * x.to_f32() + 0.5).min(1.0).max(0.0).to_f16(), |x| SimdMath::hard_sigmoid(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1,
            |x| (0.5 * x.to_f32() * (libm::erff(x.to_f32() * std::f32::consts::FRAC_1_SQRT_2) + 1.0)).to_f16(),
            |x| SimdMath::gelu(x));
        test_float_simd_math!(f16, F16Vec::SIZE, hard_swish_range.clone(), 1000, 1, |x| (x.to_f32() * (x.to_f32() + 3.0).__relu6() * (1.0 / 6.0)).to_f16(), |x| SimdMath::hard_swish(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| (1.0 + x.to_f32().exp()).ln().to_f16(), |x| SimdMath::softplus(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| (x.to_f32() / (1.0 + x.to_f32().abs())).to_f16(), |x| SimdMath::softsign(x));
        test_float_simd_math!(f16, F16Vec::SIZE, range2.clone(), 1000, 1, |x| (x.to_f32() * x.to_f32().__softplus().tanh()).to_f16(), |x| SimdMath::mish(x));
    }

    #[rustfmt::skip]
    #[test]
    fn test_int_simd_math() {
        macro_rules! test_int_simd_math {
            ($type:ty, $size:expr, $range:expr, $repeat:expr, $scalar_op:expr, $vec_op:expr) => {
                test_computes_1operands_int::<$type, { $size }>(
                    $range,
                    $repeat,
                    $scalar_op,
                    $vec_op,
                    stringify!($type::$scalar_op),
                );
            };
        }
        macro_rules! test_int_simd_template {
            ($type:ty, $vec_ty:ident, $repeat:expr) => {
                test_int_simd_math!($type, $vec_ty::SIZE, (<$type>::MIN as f64).sqrt() as $type..=(<$type>::MAX as f64).sqrt() as $type, $repeat,|x| x * x, |x| SimdMath::square(x));
                test_int_simd_math!($type, $vec_ty::SIZE, (<$type>::MIN as f64).sqrt() as $type..=(<$type>::MAX as f64).sqrt() as $type, $repeat, |x| x * x, |x| x.__square());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| x.abs(), |x| x.abs());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| x.abs(), |x| x.__abs());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.floor());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__floor());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.ceil());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__ceil());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| -x, |x| SimdMath::neg(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| -x, |x| x.__neg());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.round());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__round());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.signum(), |x| x.signum());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.signum(), |x| x.__signum());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| if x > 0 {x} else {0}, |x| SimdMath::relu(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| if x > 0 {x} else {0}, |x| x.__relu());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.max(0).min(6), |x| SimdMath::relu6(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.max(0).min(6), |x| x.__relu6());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| SimdMath::trunc(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| (x != 0) as $type, |x| x.__is_true());
            };
            (unsigned, $type:ty, $vec_ty:ident, $repeat:expr) => {
                test_int_simd_math!($type, $vec_ty::SIZE, (<$type>::MIN as f64).sqrt() as $type..=(<$type>::MAX as f64).sqrt() as $type, $repeat,|x| x * x, |x| SimdMath::square(x));
                test_int_simd_math!($type, $vec_ty::SIZE, (<$type>::MIN as f64).sqrt() as $type..=(<$type>::MAX as f64).sqrt() as $type, $repeat, |x| x * x, |x| x.__square());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| x, |x| x.abs());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN + 1..=<$type>::MAX - 1, $repeat, |x| x, |x| x.__abs());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.floor());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__floor());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.ceil());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__ceil());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.round());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| x.__round());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| if x > 0 {x} else {0}, |x| SimdMath::relu(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| if x > 0 {x} else {0}, |x| x.__relu());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.max(0).min(6), |x| SimdMath::relu6(x));
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x.max(0).min(6), |x| x.__relu6());
                test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| x, |x| SimdMath::trunc(x));
                // test_int_simd_math!($type, $vec_ty::SIZE, <$type>::MIN..=<$type>::MAX, $repeat, |x| (x != 0) as $type, |x| x.__is_true());
            };
        }
        test_int_simd_template!(i8, I8Vec, 100);
        test_int_simd_template!(i16, I16Vec, 100);
        test_int_simd_template!(i32, I32Vec, 100);
        test_int_simd_template!(i64, I64Vec, 100);
        test_int_simd_template!(isize, IsizeVec, 100);
        test_int_simd_template!(unsigned, u8, U8Vec, 100);
        test_int_simd_template!(unsigned, u16, U16Vec, 100);
        test_int_simd_template!(unsigned, u32, U32Vec, 100);
        test_int_simd_template!(unsigned, u64, U64Vec, 100);
        test_int_simd_template!(unsigned, usize, UsizeVec, 100);
    }

    #[rustfmt::skip]
    #[test]
    fn test_float_simd_math_2operands() {
        macro_rules! test_float_simd_math_2operands {
            ($type:ty, $size:expr, $lhs_range:expr, $rhs_range:expr, $repeat:expr, $scalar_op:expr, $vec_op:expr) => {
                test_computes_2operands_float::<$type, { $size }>(
                    $lhs_range,
                    $rhs_range,
                    $repeat,
                    $scalar_op,
                    $vec_op,
                    |a, b| {
                        paste::paste! {
                            [<$type _ulp_diff>](a, b) <= 1
                        }
                    },
                    stringify!($type::$scalar_op),
                );
            };
        }
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -10.0..=10.0, -2.0..=2.0, 1000, |x, y| x.powf(y), |x, y| SimdMath::pow(x, y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -10.0..=10.0, -2.0..=2.0, 1000, |x, y| x.powf(y), |x, y| x.__pow(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e37..=1e37, -1e37..=1e37, 1000, |x, y| x / y, |x, y| x.__div(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e37..=1e37, -1e37..=1e37, 1000, |x, y| x.max(0.0) + y * x.min(0.0), |x, y| SimdMath::leaky_relu(x, y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e37..=1e37, -1e37..=1e37, 1000, |x, y| x.max(0.0) + y * x.min(0.0), |x, y| x.__leaky_relu(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, 0.1..=0.9, 1.1..=1e37, 1000, |x, y| x.log(y), |x, y| x.__log(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x + y, |x, y| x.__add(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x - y, |x, y| x.__sub(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x % y, |x, y| x.__rem(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x.max(y), |x, y| x.__max(y));
        test_float_simd_math_2operands!(f32, F32Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x.min(y), |x, y| x.__min(y));

        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -100.0..=100.0, -2.0..=2.0, 1000, |x, y| x.powf(y), |x, y| SimdMath::pow(x, y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -100.0..=100.0, -2.0..=2.0, 1000, |x, y| x.powf(y), |x, y| x.__pow(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e306..=1e306, -1e306..=1e306, 1000, |x, y| x / y, |x, y| x.__div(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e306..=1e306, -1e306..=1e306, 1000, |x, y| x.max(0.0) + y * x.min(0.0), |x, y| SimdMath::leaky_relu(x, y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e306..=1e306, -1e306..=1e306, 1000, |x, y| x.max(0.0) + y * x.min(0.0), |x, y| x.__leaky_relu(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, 0.1..=0.9, 1.1..=1e306, 1000, |x, y| x.log(y), |x, y| x.__log(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x + y, |x, y| x.__add(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x - y, |x, y| x.__sub(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x % y, |x, y| x.__rem(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x.max(y), |x, y| x.__max(y));
        test_float_simd_math_2operands!(f64, F64Vec::SIZE, -1e15..=1e15, -1e15..=1e15, 1000, |x, y| x.min(y), |x, y| x.__min(y));

        use half::f16;
        let range1 = half::f16::from_f32(-4.0)..=half::f16::from_f32(4.0);
        let range2 = half::f16::from_f32(-1.0)..=half::f16::from_f32(1.0);
        let range3 = half::f16::from_f32(-1e4)..=half::f16::from_f32(1e4);
        let log_range_base = half::f16::from_f32(0.1)..=half::f16::from_f32(0.9);
        let log_range_base2 = half::f16::from_f32(1.1)..=half::f16::from_f32(1e4);
        use num_traits::real::Real;
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, range1.clone(), range2.clone(), 1000, |x, y| x.powf(y), |x, y| SimdMath::pow(x, y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, range1.clone(), range2.clone(), 1000, |x, y| x.powf(y), |x, y| x.__pow(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x / y, |x, y| x.__div(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x.max(f16::ZERO) + y * x.min(f16::ZERO), |x, y| SimdMath::leaky_relu(x, y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x.max(f16::ZERO) + y * x.min(f16::ZERO), |x, y| x.__leaky_relu(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, log_range_base.clone(), log_range_base2.clone(), 1000, |x, y| x.log(y), |x, y| x.__log(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, f16::MIN..=f16::MAX, f16::MIN..=f16::MAX, 1000, |x, y| x + y, |x, y| x.__add(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, f16::MIN..=f16::MAX, f16::MIN..=f16::MAX, 1000, |x, y| x - y, |x, y| x.__sub(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, f16::MIN..=f16::MAX, f16::MIN..=f16::MAX, 1000, |x, y| x % y, |x, y| x.__rem(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, f16::MIN..=f16::MAX, f16::MIN..=f16::MAX, 1000, |x, y| x.max(y), |x, y| x.__max(y));
        test_float_simd_math_2operands!(f16, F16Vec::SIZE, f16::MIN..=f16::MAX, f16::MIN..=f16::MAX, 1000, |x, y| x.min(y), |x, y| x.__min(y));

        use half::bf16;
        let range1 = half::bf16::from_f32(-4.0)..=half::bf16::from_f32(4.0);
        let range2 = half::bf16::from_f32(-1.0)..=half::bf16::from_f32(1.0);
        let range3 = half::bf16::from_f32(-1e4)..=half::bf16::from_f32(1e4);
        let log_range_base = half::bf16::from_f32(0.1)..=half::bf16::from_f32(0.9);
        let log_range_base2 = half::bf16::from_f32(1.1)..=half::bf16::from_f32(1e4);
        let normal_range = half::bf16::from_f32(bf16::MIN.to_f32() / 2.0)..=half::bf16::from_f32(bf16::MAX.to_f32() / 2.0);
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, range1.clone(), range2.clone(), 1000, |x, y| x.powf(y), |x, y| SimdMath::pow(x, y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, range1.clone(), range2.clone(), 1000, |x, y| x.powf(y), |x, y| x.__pow(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x / y, |x, y| x.__div(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x.max(bf16::ZERO) + y * x.min(bf16::ZERO), |x, y| SimdMath::leaky_relu(x, y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, range3.clone(), range3.clone(), 1000, |x, y| x.max(bf16::ZERO) + y * x.min(bf16::ZERO), |x, y| x.__leaky_relu(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, log_range_base.clone(), log_range_base2.clone(), 1000, |x, y| x.log(y), |x, y| x.__log(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, normal_range.clone(), normal_range.clone(), 1000, |x, y| x + y, |x, y| x.__add(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, normal_range.clone(), normal_range.clone(), 1000, |x, y| x - y, |x, y| x.__sub(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, normal_range.clone(), normal_range.clone(), 1000, |x, y| x % y, |x, y| x.__rem(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, normal_range.clone(), normal_range.clone(), 1000, |x, y| x.max(y), |x, y| x.__max(y));
        test_float_simd_math_2operands!(bf16, Bf16Vec::SIZE, normal_range.clone(), normal_range.clone(), 1000, |x, y| x.min(y), |x, y| x.__min(y));
    }

    #[rustfmt::skip]
    #[test]
    fn test_int_simd_math_2operands() {
        macro_rules! test_float_simd_math_2operands {
            ($type:ty, $size:expr, $lhs_range:expr, $rhs_range:expr, $repeat:expr, $scalar_op:expr, $vec_op:expr) => {
                test_computes_2operands_int::<$type, { $size }>(
                    $lhs_range,
                    $rhs_range,
                    $repeat,
                    $scalar_op,
                    $vec_op,
                    stringify!($type::$scalar_op),
                );
            };
        }
        macro_rules! test_int_simd_math_2operands {
            ($type: ty, $type_vec: ident) => {
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -2..=2, 0..=5, 100, |x, y| x.pow(y as u32), |x, y| SimdMath::pow(x, y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -2..=2, 0..=5, 100, |x, y| x.pow(y as u32), |x, y| x.__pow(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -63..=63, -2..=2, 100, |x, y| x.max(0) + y * x.min(0), |x, y| SimdMath::leaky_relu(x, y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -63..=63, -2..=2, 100, |x, y| x.max(0) + y * x.min(0), |x, y| x.__leaky_relu(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -63..=63, -63..=63, 100, |x, y| x + y, |x, y| x.__add(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -63..=63, -63..=63, 100, |x, y| x - y, |x, y| x.__sub(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, -63..=63, 1..=63, 100, |x, y| x % y, |x, y| x.__rem(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, <$type>::MIN..=<$type>::MAX, <$type>::MIN..=<$type>::MAX, 100, |x, y| x.max(y), |x, y| x.__max(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, <$type>::MIN..=<$type>::MAX, <$type>::MIN..=<$type>::MAX, 100, |x, y| x.min(y), |x, y| x.__min(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x >> y, |x, y| x._shr(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x << y, |x, y| x._shl(y));
            };
            (unsigned, $type: ty, $type_vec: ident) => {
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=3, 0..=5, 100, |x, y| x.pow(y as u32), |x, y| SimdMath::pow(x, y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=3, 0..=5, 100, |x, y| x.pow(y as u32), |x, y| x.__pow(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=63, 0..=2, 100, |x, y| x.max(0) + y * x.min(0), |x, y| SimdMath::leaky_relu(x, y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=63, 0..=2, 100, |x, y| x.max(0) + y * x.min(0), |x, y| x.__leaky_relu(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=63, 0..=63, 100, |x, y| x + y, |x, y| x.__add(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 30..=63, 0..=30, 100, |x, y| x - y, |x, y| x.__sub(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=63, 1..=63, 100, |x, y| x % y, |x, y| x.__rem(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, <$type>::MIN..=<$type>::MAX, <$type>::MIN..=<$type>::MAX, 100, |x, y| x.max(y), |x, y| x.__max(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, <$type>::MIN..=<$type>::MAX, <$type>::MIN..=<$type>::MAX, 100, |x, y| x.min(y), |x, y| x.__min(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x >> y, |x, y| x._shr(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x << y, |x, y| x._shl(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x | y, |x, y| x._bitor(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x & y, |x, y| x._bitand(y));
                test_float_simd_math_2operands!($type, $type_vec::SIZE, 0..=10, 0..=2, 100, |x, y| x ^ y, |x, y| x._bitxor(y));
            };
        }
        test_int_simd_math_2operands!(i8, I8Vec);
        test_int_simd_math_2operands!(i16, I16Vec);
        test_int_simd_math_2operands!(i32, I32Vec);
        test_int_simd_math_2operands!(i64, I64Vec);
        test_int_simd_math_2operands!(isize, IsizeVec);
        test_int_simd_math_2operands!(unsigned, u8, U8Vec);
        test_int_simd_math_2operands!(unsigned, u16, U16Vec);
        test_int_simd_math_2operands!(unsigned, u32, U32Vec);
        test_int_simd_math_2operands!(unsigned, u64, U64Vec);
        test_int_simd_math_2operands!(unsigned, usize, UsizeVec);
    }

    #[test]
    fn test_is_infinite() {
        macro_rules! test_is_infinite {
            ($type: ty, $vec_ty: ident, $mask_type: ident) => {
                let arr = [<$type>::INFINITY; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([1; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of inf is_inf));
                let arr = [<$type>::NAN; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of nan is_inf));
                let arr = [<$type>::NEG_INFINITY; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([-1; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of -inf is_inf));
            };
            (2, $type: ty, $vec_ty: ident, $mask_type: ident) => {
                let arr = [<$type>::MAX; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of inf is_inf));
                let arr = [<$type>::ZERO; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of nan is_inf));
                let arr = [<$type>::MIN; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_inf(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of -inf is_inf));
            };
        }
        test_is_infinite!(f32, F32Vec, I32Vec);
        test_is_infinite!(f64, F64Vec, I64Vec);
        test_is_infinite!(half::f16, F16Vec, I16Vec);
        test_is_infinite!(half::bf16, Bf16Vec, I16Vec);
        test_is_infinite!(2, i8, I8Vec, I8Vec);
        test_is_infinite!(2, i16, I16Vec, I16Vec);
        test_is_infinite!(2, i32, I32Vec, I32Vec);
        test_is_infinite!(2, i64, I64Vec, I64Vec);
        test_is_infinite!(2, u8, U8Vec, I8Vec);
        test_is_infinite!(2, u16, U16Vec, I16Vec);
        test_is_infinite!(2, u32, U32Vec, I32Vec);
        test_is_infinite!(2, u64, U64Vec, I64Vec);
        test_is_infinite!(2, usize, UsizeVec, IsizeVec);
        test_is_infinite!(2, isize, IsizeVec, IsizeVec);
    }

    #[test]
    fn test_is_nan() {
        macro_rules! test_is_nan {
            ($type: ty, $vec_ty: ident, $mask_type: ident) => {
                let arr = [<$type>::INFINITY; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of inf is_nan));
                let arr = [<$type>::NAN; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([-1; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of nan is_nan));
                let arr = [<$type>::NEG_INFINITY; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of -inf is_nan));
            };
            (2, $type: ty, $vec_ty: ident, $mask_type: ident) => {
                let arr = [<$type>::MAX; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of inf is_nan));
                let arr = [<$type>::ZERO; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of nan is_nan));
                let arr = [<$type>::MIN; $vec_ty::SIZE];
                let a = unsafe { $vec_ty::from_ptr(arr.as_ptr()) };
                assert_eq!(a.__is_nan(), unsafe {
                    $mask_type::from_ptr([0; $vec_ty::SIZE].as_ptr())
                }, stringify!($type of -inf is_nan));
            };
        }
        test_is_nan!(f32, F32Vec, I32Vec);
        test_is_nan!(f64, F64Vec, I64Vec);
        test_is_nan!(half::f16, F16Vec, I16Vec);
        test_is_nan!(half::bf16, Bf16Vec, I16Vec);
        test_is_nan!(2, i8, I8Vec, I8Vec);
        test_is_nan!(2, i16, I16Vec, I16Vec);
        test_is_nan!(2, i32, I32Vec, I32Vec);
        test_is_nan!(2, i64, I64Vec, I64Vec);
        test_is_nan!(2, u8, U8Vec, I8Vec);
        test_is_nan!(2, u16, U16Vec, I16Vec);
        test_is_nan!(2, u32, U32Vec, I32Vec);
        test_is_nan!(2, u64, U64Vec, I64Vec);
        test_is_nan!(2, usize, UsizeVec, IsizeVec);
        test_is_nan!(2, isize, IsizeVec, IsizeVec);
    }

    #[test]
    fn test_sum() {}
}
