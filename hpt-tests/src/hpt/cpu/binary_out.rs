#![allow(unused_imports)]
use hpt::*;
use hpt_common::slice;
use hpt_macros::select;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::*;
use tch::Tensor as TchTensor;

#[allow(unused)]
fn assert_eq(b: &Tensor<f64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let rel_diff = if a_raw[i] == 0.0 && b_raw[i] == 0.0 {
            0.0
        } else {
            abs_diff / (a_raw[i].abs() + b_raw[i].abs() + f64::EPSILON)
        };

        if rel_diff > 0.05 {
            panic!("{} != {} (relative_diff: {})", a_raw[i], b_raw[i], rel_diff);
        }
    }
}

#[allow(unused)]
fn assert_eq_10(b: &Tensor<f64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let rel_diff = if a_raw[i] == 0.0 && b_raw[i] == 0.0 {
            0.0
        } else {
            abs_diff / (a_raw[i].abs() + b_raw[i].abs() + f64::EPSILON)
        };

        if rel_diff > 0.05 {
            panic!("{} != {} (relative_diff: {})", a_raw[i], b_raw[i], rel_diff);
        }
    }
}

#[allow(unused)]
fn assert_eq_i64(b: &Tensor<i64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    a_raw
        .par_iter()
        .zip(b_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

#[allow(unused)]
fn assert_eq_bool(b: &Tensor<bool>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const bool, b.size()) };
    let b_raw = b.as_raw();
    a_raw
        .par_iter()
        .zip(b_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

#[allow(unused)]
fn no_assert_i64(b: &Tensor<i64>, a: &TchTensor) {}

#[allow(unused)]
fn common_input<const N: usize, const M: usize>(
    lhs_shape: [i64; N],
    rhs_shape: [i64; M],
) -> anyhow::Result<(
    (TchTensor, TchTensor),
    (hpt::tensor::Tensor<f64>, hpt::tensor::Tensor<f64>),
)> {
    let tch_a = TchTensor::randn(&lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut a = hpt::tensor::Tensor::<f64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });

    let tch_b = TchTensor::randn(&rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut b = hpt::tensor::Tensor::<f64>::empty(&rhs_shape)?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b_size)
    });

    Ok(((tch_a, tch_b), (a, b)))
}

#[allow(unused)]
fn common_input_i64<const N: usize, const M: usize>(
    lhs_shape: [i64; N],
    rhs_shape: [i64; M],
) -> anyhow::Result<(
    (TchTensor, TchTensor),
    (hpt::tensor::Tensor<i64>, hpt::tensor::Tensor<i64>),
)> {
    let tch_a = TchTensor::arange(
        lhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&lhs_shape);
    let mut a = hpt::tensor::Tensor::<i64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const i64, a_size)
    });

    let tch_b = TchTensor::arange(
        rhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&rhs_shape);
    let mut b = hpt::tensor::Tensor::<i64>::empty(&rhs_shape)?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const i64, b_size)
    });

    Ok(((tch_a, tch_b), (a, b)))
}

macro_rules! test_binarys {
    (
        $name:ident,
        $tch_op:ident,
        $hpt_op:ident,
        $input_method:ident,
        $assert_method:ident $(, $try:tt)*
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let mut empty = Tensor::<f64>::empty(&[10, 10])?;
                let c = a.$hpt_op(&b, &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                let c = a.$hpt_op(b, &mut empty)$($try)*;
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _broadcast>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 1])?;
                let mut empty = Tensor::<f64>::empty(&[10, 10])?;
                let c = a.$hpt_op(&b, &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                let c = a.$hpt_op(b, &mut empty)$($try)*;
                $assert_method(&c, &tch_c);

                let ((tch_a, tch_b), (a, b)) = $input_method([1, 10], [10, 1])?;
                let c = a.$hpt_op(&b, &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                let c = a.$hpt_op(b, &mut empty)$($try)*;
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _sub_tensors>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let a = slice!(a[2:6:1, 2:6:1])?;
                let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let b = slice!(b[2:6:1, 2:6:1])?;
                let empty = Tensor::<f64>::empty(&[4, 4])?;
                let c = a.$hpt_op(&b, empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _uncontiguous>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.permute(&[1, 0][..]);
                let a = a.permute([1, 0])?;
                let tch_b = tch_b.permute(&[1, 0][..]);
                let b = b.permute([1, 0])?;
                let mut empty = Tensor::<f64>::empty(&[10, 10])?;
                let c = a.$hpt_op(&b, &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _uncontiguous_sub_tensors>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let a = slice!(a[2:6:1, 2:6:1])?;
                let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let b = slice!(b[2:6:1, 2:6:1])?;
                let tch_a = tch_a.permute(&[1, 0][..]);
                let a = a.permute([1, 0])?;
                let tch_b = tch_b.permute(&[1, 0][..]);
                let b = b.permute([1, 0])?;
                let mut empty = Tensor::<f64>::empty(&[4, 4])?;
                let c = a.$hpt_op(&b, &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                Ok(())
            }
        }
    };
}

macro_rules! test_binarys_scalar {
    (
        $name:ident,
        $tch_op:ident,
        $hpt_op:ident,
        $scalar:literal,
        $ty:ident,
        $input_method:ident,
        $assert_method:ident $(, $try:tt)*
    ) => {
        paste::paste! {
            #[test]
            fn [<test _ $name>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let mut empty = hpt::tensor::Tensor::<$ty>::empty(&[10, 10])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(TchTensor::from($scalar));
                $assert_method(&c, &tch_c);
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _broadcast>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 1])?;
                let mut empty = hpt::tensor::Tensor::<$ty>::empty(&[10, 10])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(TchTensor::from($scalar));
                $assert_method(&c, &tch_c);
                let mut empty = hpt::tensor::Tensor::<$ty>::empty(&[10])?;
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b);
                $assert_method(&c, &tch_c);

                let ((tch_a, tch_b), (a, b)) = $input_method([1, 10], [10, 1])?;
                let mut empty = hpt::tensor::Tensor::<$ty>::empty(&[10])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(TchTensor::from($scalar));
                $assert_method(&c, &tch_c);
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _sub_tensors>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let a = slice!(a[2:6:1, 2:6:1])?;
                let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let b = slice!(b[2:6:1, 2:6:1])?;
                let mut empty = hpt::tensor::Tensor::<f64>::empty(&[4, 4])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.shallow_clone().$tch_op(TchTensor::from($scalar));
                $assert_method(&c, &tch_c);
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _uncontiguous>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.permute(&[1, 0][..]);
                let a = a.permute([1, 0])?;
                let tch_b = tch_b.permute(&[1, 0][..]);
                let b = b.permute([1, 0])?;
                let mut empty = hpt::tensor::Tensor::<f64>::empty(&[10, 10])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(TchTensor::from($scalar)).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _uncontiguous_sub_tensors>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 10])?;
                let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let a = slice!(a[2:6:1, 2:6:1])?;
                let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
                let b = slice!(b[2:6:1, 2:6:1])?;
                let tch_a = tch_a.permute(&[1, 0][..]);
                let a = a.permute([1, 0])?;
                let tch_b = tch_b.permute(&[1, 0][..]);
                let b = b.permute([1, 0])?;
                let mut empty = hpt::tensor::Tensor::<f64>::empty(&[4, 4])?;
                let c = a.$hpt_op(&hpt::tensor::Tensor::new($scalar), &mut empty)$($try)*;
                let tch_c = tch_a.$tch_op(TchTensor::from($scalar)).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                let c = hpt::tensor::Tensor::new($scalar).$hpt_op(b, &mut empty)$($try)*;
                let tch_c = TchTensor::from($scalar).$tch_op(&tch_b).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                Ok(())
            }
        }
    };
}

test_binarys!(add, add, add_, common_input, assert_eq, ?);
test_binarys!(sub, sub, sub_, common_input, assert_eq, ?);
test_binarys!(mul, mul, mul_, common_input, assert_eq, ?);
test_binarys!(rem, fmod_tensor, rem_, common_input, assert_eq, ?);
// test_binarys!(div, div, div_, common_input, assert_eq);
// test_binarys!(
//     bitand,
//     bitwise_and_tensor,
//     bitand,
//     common_input_i64,
//     assert_eq_i64
// );
// test_binarys!(
//     bitor,
//     bitwise_or_tensor,
//     bitor,
//     common_input_i64,
//     assert_eq_i64
// );
// test_binarys!(
//     bitxor,
//     bitwise_xor_tensor,
//     bitxor,
//     common_input_i64,
//     assert_eq_i64
// );
// test_binarys!(
//     shl,
//     bitwise_left_shift,
//     shl,
//     common_input_i64,
//     no_assert_i64
// );
// test_binarys!(
//     shr,
//     bitwise_right_shift,
//     shr,
//     common_input_i64,
//     no_assert_i64
// );
// test_binarys!(eq, eq_tensor, tensor_eq, common_input, assert_eq_bool, ?);
// test_binarys!(ne, ne_tensor, tensor_neq, common_input, assert_eq_bool, ?);
// test_binarys!(lt, lt_tensor, tensor_lt, common_input, assert_eq_bool, ?);
// test_binarys!(le, le_tensor, tensor_le, common_input, assert_eq_bool, ?);
// test_binarys!(gt, gt_tensor, tensor_gt, common_input, assert_eq_bool, ?);
// test_binarys!(ge, ge_tensor, tensor_ge, common_input, assert_eq_bool, ?);
test_binarys!(matmul, matmul, matmul_, common_input, assert_eq_10, ?);
test_binarys_scalar!(add_scalar, add, add_, 1.0, f64, common_input, assert_eq, ?);

#[test]
fn test_binary_out_invalid_empty() -> anyhow::Result<()> {
    let (_, (a, b)) = common_input([10, 10], [10, 10])?;
    let mut empty = Tensor::<f64>::empty(&[10])?;
    let err = a.add_(&b, &mut empty).unwrap_err();
    assert!(err
        .to_string()
        .contains("Size mismatch: expected 100, got 10"));
    let err = hpt::tensor::Tensor::new(1.0f64)
        .add_(b, &mut empty)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("Size mismatch: expected 100, got 10"));
    let err = a
        .add_(hpt::tensor::Tensor::new(1.0f64), &mut empty)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("Size mismatch: expected 100, got 10"));
    Ok(())
}
