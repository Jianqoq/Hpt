#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::*;
use tch::Tensor as TchTensor;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::Matmul;
use tensor_dyn::Random;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorCmp;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{Tensor, TensorCreator};
use tensor_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &Tensor<f64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;

    for i in 0..b.size() {
        let rel_diff =
            ((a_raw[i] - b_raw[i]) / (a_raw[i].abs() + b_raw[i].abs() + f64::EPSILON)).abs();
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
) -> anyhow::Result<((TchTensor, TchTensor), (Tensor<f64>, Tensor<f64>))> {
    let tch_a = TchTensor::randn(&lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });

    let tch_b = TchTensor::randn(&rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut b = Tensor::<f64>::empty(&rhs_shape)?;
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
) -> anyhow::Result<((TchTensor, TchTensor), (Tensor<i64>, Tensor<i64>))> {
    let tch_a = TchTensor::arange(
        lhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&lhs_shape);
    let mut a = Tensor::<i64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const i64, a_size)
    });

    let tch_b = TchTensor::arange(
        rhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(&rhs_shape);
    let mut b = Tensor::<i64>::empty(&rhs_shape)?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const i64, b_size)
    });

    Ok(((tch_a, tch_b), (a, b)))
}

#[duplicate::duplicate_item(
    fn_name         hpt_op                tch_op                assert_method       input_method;
    [test_add]      [a.add(&b)]           [add]                 [assert_eq]         [common_input];
    [test_sub]      [a.sub(&b)]           [sub]                 [assert_eq]         [common_input];
    [test_mul]      [a.mul(&b)]           [mul]                 [assert_eq]         [common_input];
    [test_div]      [a.div(&b)]           [div]                 [assert_eq]         [common_input];
    [test_bitand]   [a.bitand(&b)]        [bitwise_and_tensor]  [assert_eq_i64]     [common_input_i64];
    [test_bitor]     [a.bitor(&b)]        [bitwise_or_tensor]   [assert_eq_i64]     [common_input_i64];
    [test_bitxor]   [a.bitxor(&b)]        [bitwise_xor_tensor]  [assert_eq_i64]     [common_input_i64];
    [test_shl]      [a.shl(&b)]           [bitwise_left_shift]  [no_assert_i64]     [common_input_i64];
    [test_shr]      [a.shr(&b)]           [bitwise_right_shift] [no_assert_i64]     [common_input_i64];
    [test_eq]       [a.tensor_eq(&b)?]    [eq_tensor]           [assert_eq_bool]    [common_input];
    [test_ne]       [a.tensor_neq(&b)?]   [ne_tensor]           [assert_eq_bool]    [common_input];
    [test_lt]       [a.tensor_lt(&b)?]    [lt_tensor]           [assert_eq_bool]    [common_input];
    [test_le]       [a.tensor_le(&b)?]    [le_tensor]           [assert_eq_bool]    [common_input];
    [test_gt]       [a.tensor_gt(&b)?]    [gt_tensor]           [assert_eq_bool]    [common_input];
    [test_ge]       [a.tensor_ge(&b)?]    [ge_tensor]           [assert_eq_bool]    [common_input];
    [test_matmul]   [a.matmul(&b)?]       [matmul]              [assert_eq_10]      [common_input];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = input_method([13, 13], [13, 13])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                   hpt_op                  tch_op                  assert_method       input_method;
    [test_add_broadcast]            [a.add(&b)]             [add]                   [assert_eq]         [common_input];
    [test_sub_broadcast]            [a.sub(&b)]             [sub]                   [assert_eq]         [common_input];
    [test_mul_broadcast]            [a.mul(&b)]             [mul]                   [assert_eq]         [common_input];
    [test_div_broadcast]            [a.div(&b)]             [div]                   [assert_eq]         [common_input];
    [test_bitand_broadcast]         [a.bitand(&b)]          [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_broadcast]          [a.bitor(&b)]           [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_broadcast]         [a.bitxor(&b)]          [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_broadcast]            [a.shl(&b)]             [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_broadcast]            [a.shr(&b)]             [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
    [test_eq_broadcast]             [a.tensor_eq(&b)?]      [eq_tensor]             [assert_eq_bool]    [common_input];
    [test_ne_broadcast]             [a.tensor_neq(&b)?]     [ne_tensor]             [assert_eq_bool]    [common_input];
    [test_lt_broadcast]             [a.tensor_lt(&b)?]      [lt_tensor]             [assert_eq_bool]    [common_input];
    [test_le_broadcast]             [a.tensor_le(&b)?]      [le_tensor]             [assert_eq_bool]    [common_input];
    [test_gt_broadcast]             [a.tensor_gt(&b)?]      [gt_tensor]             [assert_eq_bool]    [common_input];
    [test_ge_broadcast]             [a.tensor_ge(&b)?]      [ge_tensor]             [assert_eq_bool]    [common_input];
    [test_matmul_broadcast]         [a.matmul(&b)?]         [matmul]                [assert_eq_10]      [common_input];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = input_method([13, 13], [13, 1])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b);
    assert_method(&c, &tch_c);
    let ((tch_a, tch_b), (a, b)) = input_method([1, 13], [13, 1])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                 hpt_op                  tch_op                  assert_method       input_method;
    [test_add_sub_tensors]          [a.add(&b)]             [add]                   [assert_eq]         [common_input];
    [test_sub_sub_tensors]          [a.sub(&b)]             [sub]                   [assert_eq]         [common_input];
    [test_mul_sub_tensors]          [a.mul(&b)]             [mul]                   [assert_eq]         [common_input];
    [test_div_sub_tensors]          [a.div(&b)]             [div]                   [assert_eq]         [common_input];
    [test_bitand_sub_tensors]       [a.bitand(&b)]          [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_sub_tensors]        [a.bitor(&b)]           [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_sub_tensors]       [a.bitxor(&b)]          [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_sub_tensors]          [a.shl(&b)]             [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_sub_tensors]          [a.shr(&b)]             [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
    [test_eq_sub_tensors]           [a.tensor_eq(&b)?]      [eq_tensor]             [assert_eq_bool]    [common_input];
    [test_ne_sub_tensors]           [a.tensor_neq(&b)?]     [ne_tensor]             [assert_eq_bool]    [common_input];
    [test_lt_sub_tensors]           [a.tensor_lt(&b)?]      [lt_tensor]             [assert_eq_bool]    [common_input];
    [test_le_sub_tensors]           [a.tensor_le(&b)?]      [le_tensor]             [assert_eq_bool]    [common_input];
    [test_gt_sub_tensors]           [a.tensor_gt(&b)?]      [gt_tensor]             [assert_eq_bool]    [common_input];
    [test_ge_sub_tensors]           [a.tensor_ge(&b)?]      [ge_tensor]             [assert_eq_bool]    [common_input];
    [test_matmul_sub_tensors]       [a.matmul(&b)?]         [matmul]                [assert_eq_10]      [common_input];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let b = slice!(b[2:6:1,2:6:1])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                   hpt_op            tch_op                  assert_method       input_method;
    [test_add_uc]           [a.add(&b)]         [add]                   [assert_eq]         [common_input];
    [test_sub_uc]           [a.sub(&b)]         [sub]                   [assert_eq]         [common_input];
    [test_mul_uc]           [a.mul(&b)]         [mul]                   [assert_eq]         [common_input];
    [test_div_uc]           [a.div(&b)]         [div]                   [assert_eq]         [common_input];
    [test_bitand_uc]        [a.bitand(&b)]      [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_uc]         [a.bitor(&b)]       [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_uc]        [a.bitxor(&b)]      [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_uc]           [a.shl(&b)]         [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_uc]           [a.shr(&b)]         [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
    [test_eq_uc]            [a.tensor_eq(&b)?]  [eq_tensor]             [assert_eq_bool]    [common_input];
    [test_ne_uc]            [a.tensor_neq(&b)?] [ne_tensor]             [assert_eq_bool]    [common_input];
    [test_lt_uc]            [a.tensor_lt(&b)?]  [lt_tensor]             [assert_eq_bool]    [common_input];
    [test_le_uc]            [a.tensor_le(&b)?]  [le_tensor]             [assert_eq_bool]    [common_input];
    [test_gt_uc]            [a.tensor_gt(&b)?]  [gt_tensor]             [assert_eq_bool]    [common_input];
    [test_ge_uc]            [a.tensor_ge(&b)?]  [ge_tensor]             [assert_eq_bool]    [common_input];
    [test_matmul_uc]        [a.matmul(&b)?]     [matmul]                [assert_eq_10]      [common_input];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let tch_b = tch_b.permute(&[1, 0][..]);
    let b = b.permute([1, 0])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b).contiguous();
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                             hpt_op                  tch_op                  assert_method       input_method;
    [test_add_uc_sub_tensors]           [a.add(&b)]             [add]                   [assert_eq]         [common_input];
    [test_sub_uc_sub_tensors]           [a.sub(&b)]             [sub]                   [assert_eq]         [common_input];
    [test_mul_uc_sub_tensors]           [a.mul(&b)]             [mul]                   [assert_eq]         [common_input];
    [test_div_uc_sub_tensors]           [a.div(&b)]             [div]                   [assert_eq]         [common_input];
    [test_bitand_uc_sub_tensors]        [a.bitand(&b)]          [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_uc_sub_tensors]         [a.bitor(&b)]           [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_uc_sub_tensors]        [a.bitxor(&b)]          [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_uc_sub_tensors]           [a.shl(&b)]             [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_uc_sub_tensors]           [a.shr(&b)]             [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
    [test_eq_uc_sub_tensors]            [a.tensor_eq(&b)?]      [eq_tensor]             [assert_eq_bool]    [common_input];
    [test_ne_uc_sub_tensors]            [a.tensor_neq(&b)?]     [ne_tensor]             [assert_eq_bool]    [common_input];
    [test_lt_uc_sub_tensors]            [a.tensor_lt(&b)?]      [lt_tensor]             [assert_eq_bool]    [common_input];
    [test_le_uc_sub_tensors]            [a.tensor_le(&b)?]      [le_tensor]             [assert_eq_bool]    [common_input];
    [test_gt_uc_sub_tensors]            [a.tensor_gt(&b)?]      [gt_tensor]             [assert_eq_bool]    [common_input];
    [test_ge_uc_sub_tensors]            [a.tensor_ge(&b)?]      [ge_tensor]             [assert_eq_bool]    [common_input];
    [test_matmul_uc_sub_tensors]        [a.matmul(&b)?]         [matmul]                [assert_eq_10]      [common_input];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let b = slice!(b[2:6:1,2:6:1])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let tch_b = tch_b.permute(&[1, 0][..]);
    let b = b.permute([1, 0])?;
    let c = hpt_op;
    let tch_c = tch_a.tch_op(&tch_b).contiguous();
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                  scalar    hpt_op       tch_op                  assert_method       input_method;
    [test_add_scalar]        [1.0]     [add]        [add]                   [assert_eq]         [common_input];
    [test_sub_scalar]        [1.0]     [sub]        [sub]                   [assert_eq]         [common_input];
    [test_mul_scalar]        [1.0]     [mul]        [mul]                   [assert_eq]         [common_input];
    [test_div_scalar]        [1.0]     [div]        [div]                   [assert_eq]         [common_input];
    [test_bitand_scalar]     [1]       [bitand]     [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_scalar]      [1]       [bitor]      [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_scalar]     [1]       [bitxor]     [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_scalar]        [1]       [shl]        [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_scalar]        [1]       [shr]        [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, _), (a, _)) = input_method([13, 13], [13, 13])?;
    let c = a.clone().hpt_op(scalar);
    let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
              fn_name                  scalar    hpt_op         tch_op                  assert_method       input_method;
    [test_add_scalar_broadcast]        [1.0]     [add]          [add]                   [assert_eq]         [common_input];
    [test_sub_scalar_broadcast]        [1.0]     [sub]          [sub]                   [assert_eq]         [common_input];
    [test_mul_scalar_broadcast]        [1.0]     [mul]          [mul]                   [assert_eq]         [common_input];
    [test_div_scalar_broadcast]        [1.0]     [div]          [div]                   [assert_eq]         [common_input];
    [test_bitand_scalar_broadcast]     [1]       [bitand]       [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_scalar_broadcast]      [1]       [bitor]        [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_scalar_broadcast]     [1]       [bitxor]       [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_scalar_broadcast]        [1]       [shl]          [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_scalar_broadcast]        [1]       [shr]          [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, _), (a, _)) = input_method([13, 13], [13, 1])?;
    let c = a.clone().add(scalar);
    let tch_c = tch_a.shallow_clone().add(&TchTensor::from(scalar));
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
    assert_method(&c, &tch_c);
    let ((tch_a, _), (a, _)) = input_method([1, 13], [13, 1])?;
    let c = a.clone().hpt_op(scalar);
    let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                              scalar    hpt_op       tch_op                  assert_method       input_method;
    [test_add_scalar_sub_tensors]        [1.0]     [add]        [add]                   [assert_eq]         [common_input];
    [test_sub_scalar_sub_tensors]        [1.0]     [sub]        [sub]                   [assert_eq]         [common_input];
    [test_mul_scalar_sub_tensors]        [1.0]     [mul]        [mul]                   [assert_eq]         [common_input];
    [test_div_scalar_sub_tensors]        [1.0]     [div]        [div]                   [assert_eq]         [common_input];
    [test_bitand_scalar_sub_tensors]     [1]       [bitand]     [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_scalar_sub_tensors]      [1]       [bitor]      [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_scalar_sub_tensors]     [1]       [bitxor]     [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_scalar_sub_tensors]        [1]       [shl]        [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_scalar_sub_tensors]        [1]       [shr]        [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, _), (a, _)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let c = a.clone().hpt_op(scalar);
    let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                     scalar    hpt_op        tch_op                  assert_method       input_method;
    [test_add_scalar_uc]        [1.0]     [add]         [add]                   [assert_eq]         [common_input];
    [test_sub_scalar_uc]        [1.0]     [sub]         [sub]                   [assert_eq]         [common_input];
    [test_mul_scalar_uc]        [1.0]     [mul]         [mul]                   [assert_eq]         [common_input];
    [test_div_scalar_uc]        [1.0]     [div]         [div]                   [assert_eq]         [common_input];
    [test_bitand_scalar_uc]     [1]       [bitand]      [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_bitor_scalar_uc]      [1]       [bitor]       [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
    [test_bitxor_scalar_uc]     [1]       [bitxor]      [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
    [test_shl_scalar_uc]        [1]       [shl]         [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
    [test_shr_scalar_uc]        [1]       [shr]         [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, _), (a, _)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let c = a.clone().hpt_op(scalar);
    let tch_c = tch_a
        .shallow_clone()
        .tch_op(&TchTensor::from(scalar))
        .contiguous();
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a).contiguous();
    assert_method(&c, &tch_c);
    Ok(())
}

#[duplicate::duplicate_item(
    fn_name                             scalar    hpt_op        tch_op                  assert_method       input_method;
[test_add_scalar_uc_sub_tensors]        [1.0]     [add]         [add]                   [assert_eq]         [common_input];
[test_sub_scalar_uc_sub_tensors]        [1.0]     [sub]         [sub]                   [assert_eq]         [common_input];
[test_mul_scalar_uc_sub_tensors]        [1.0]     [mul]         [mul]                   [assert_eq]         [common_input];
[test_div_scalar_uc_sub_tensors]        [1.0]     [div]         [div]                   [assert_eq]         [common_input];
[test_bitand_scalar_uc_sub_tensors]     [1]       [bitand]      [bitwise_and_tensor]    [assert_eq_i64]     [common_input_i64];
[test_bitor_scalar_uc_sub_tensors]      [1]       [bitor]       [bitwise_or_tensor]     [assert_eq_i64]     [common_input_i64];
[test_bitxor_scalar_uc_sub_tensors]     [1]       [bitxor]      [bitwise_xor_tensor]    [assert_eq_i64]     [common_input_i64];
[test_shl_scalar_uc_sub_tensors]        [1]       [shl]         [bitwise_left_shift]    [no_assert_i64]     [common_input_i64];
[test_shr_scalar_uc_sub_tensors]        [1]       [shr]         [bitwise_right_shift]   [no_assert_i64]     [common_input_i64];
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let ((tch_a, _), (a, _)) = input_method([13, 13], [13, 13])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let c = a.clone().hpt_op(scalar);
    let tch_c = tch_a
        .shallow_clone()
        .tch_op(&TchTensor::from(scalar))
        .contiguous();
    assert_method(&c, &tch_c);
    let c = scalar.hpt_op(&a);
    let tch_c = TchTensor::from(scalar).tch_op(&tch_a).contiguous();
    assert_method(&c, &tch_c);
    Ok(())
}

#[test]
fn test_batch_matmul() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([13, 13, 13], [13, 13, 13])?;
    let c = a.matmul(&b)?;
    let tch_c = tch_a.matmul(&tch_b);
    assert_eq(&c, &tch_c);
    Ok(())
}

#[should_panic(expected = "should panic")]
#[test]
fn test_batch_matmul_panic() {
    let a = Tensor::<f64>::empty([13, 4]).expect("should not panic");
    let b = Tensor::<f64>::empty([3, 13]).expect("should not panic");
    a.matmul(&b).expect("should panic");
}
