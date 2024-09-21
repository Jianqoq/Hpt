#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::*;
use tch::Tensor;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::Matmul;
use tensor_dyn::Random;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorCmp;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{tensor_base::_Tensor, TensorCreator};
use tensor_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &_Tensor<f64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            panic!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a_raw[i], b_raw[i], abs_diff, relative_diff
            );
        }
    }
}

#[allow(unused)]
fn assert_eq_10(b: &_Tensor<f64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 10e-16;

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let relative_diff = abs_diff / b_raw[i].abs().max(f64::EPSILON);

        if abs_diff > tolerance && relative_diff > tolerance {
            panic!(
                "{} != {} (abs_diff: {}, relative_diff: {})",
                a_raw[i], b_raw[i], abs_diff, relative_diff
            );
        }
    }
}

#[allow(unused)]
fn assert_eq_i64(b: &_Tensor<i64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    a_raw
        .par_iter()
        .zip(b_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

#[allow(unused)]
fn assert_eq_bool(b: &_Tensor<bool>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const bool, b.size()) };
    let b_raw = b.as_raw();
    a_raw
        .par_iter()
        .zip(b_raw.par_iter())
        .for_each(|(a, b)| assert_eq!(a, b));
}

#[allow(unused)]
fn no_assert_i64(b: &_Tensor<i64>, a: &Tensor) {}

#[allow(unused)]
fn common_input<const N: usize, const M: usize>(
    lhs_shape: [i64; N],
    rhs_shape: [i64; M],
) -> anyhow::Result<((Tensor, Tensor), (_Tensor<f64>, _Tensor<f64>))> {
    let tch_a = Tensor::randn(&lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut a = _Tensor::<f64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });

    let tch_b = Tensor::randn(&rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut b = _Tensor::<f64>::empty(&rhs_shape)?;
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
) -> anyhow::Result<((Tensor, Tensor), (_Tensor<i64>, _Tensor<i64>))> {
    let tch_a = Tensor::arange(
        lhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&lhs_shape);
    let mut a = _Tensor::<i64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const i64, a_size)
    });

    let tch_b = Tensor::arange(
        rhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&rhs_shape);
    let mut b = _Tensor::<i64>::empty(&rhs_shape)?;
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
                let c = a.$hpt_op(&b)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);
                Ok(())
            }

            #[test]
            fn [<test_ $name _broadcast>]() -> anyhow::Result<()> {
                let ((tch_a, tch_b), (a, b)) = $input_method([10, 10], [10, 1])?;
                let c = a.$hpt_op(&b)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
                $assert_method(&c, &tch_c);

                let ((tch_a, tch_b), (a, b)) = $input_method([1, 10], [10, 1])?;
                let c = a.$hpt_op(&b)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b);
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
                let c = a.$hpt_op(&b)$($try)*;
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
                let c = a.$hpt_op(&b)$($try)*;
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
                let c = a.$hpt_op(&b)$($try)*;
                let tch_c = tch_a.$tch_op(&tch_b).contiguous(); // torch will keep the layout, so we need contiguous
                $assert_method(&c, &tch_c);
                Ok(())
            }
        }
    };
}

test_binarys!(add, add, add, common_input, assert_eq);
test_binarys!(sub, sub, sub, common_input, assert_eq);
test_binarys!(mul, mul, mul, common_input, assert_eq);
test_binarys!(div, div, div, common_input, assert_eq);
test_binarys!(
    bitand,
    bitwise_and_tensor,
    bitand,
    common_input_i64,
    assert_eq_i64
);
test_binarys!(
    bitor,
    bitwise_or_tensor,
    bitor,
    common_input_i64,
    assert_eq_i64
);
test_binarys!(
    bitxor,
    bitwise_xor_tensor,
    bitxor,
    common_input_i64,
    assert_eq_i64
);
test_binarys!(
    shl,
    bitwise_left_shift,
    shl,
    common_input_i64,
    no_assert_i64
);
test_binarys!(
    shr,
    bitwise_right_shift,
    shr,
    common_input_i64,
    no_assert_i64
);
test_binarys!(eq, eq_tensor, tensor_eq, common_input, assert_eq_bool, ?);
test_binarys!(ne, ne_tensor, tensor_neq, common_input, assert_eq_bool, ?);
test_binarys!(lt, lt_tensor, tensor_lt, common_input, assert_eq_bool, ?);
test_binarys!(le, le_tensor, tensor_le, common_input, assert_eq_bool, ?);
test_binarys!(gt, gt_tensor, tensor_gt, common_input, assert_eq_bool, ?);
test_binarys!(ge, ge_tensor, tensor_ge, common_input, assert_eq_bool, ?);
test_binarys!(matmul, matmul, matmul, common_input, assert_eq_10, ?);
