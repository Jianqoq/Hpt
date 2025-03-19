#![allow(unused_imports)]
use super::assert_utils::assert_f64;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Matmul;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCmp;
use hpt::ops::TensorCreator;
use hpt::*;
use hpt_common::slice;
use hpt_macros::select;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::*;
use tch::Tensor as TchTensor;

#[allow(unused)]
fn assert_eq(b: &Tensor<f64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();
    let tolerance = 2.5e-16;

    for i in 0..b.size() {
        assert_f64(a_raw[i], b_raw[i], 0.05, b, a);
    }
}

#[allow(unused)]
fn assert_eq_10(b: &Tensor<f64>, a: &TchTensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const f64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        assert_f64(a_raw[i], b_raw[i], 0.10, b, a);
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
fn common_input(
    lhs_shape: &[i64],
    rhs_shape: &[i64],
) -> anyhow::Result<((TchTensor, TchTensor), (Tensor<f64>, Tensor<f64>))> {
    let tch_a = TchTensor::randn(lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut a = Tensor::<f64>::empty(lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });

    let tch_b = TchTensor::randn(rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut b = Tensor::<f64>::empty(rhs_shape)?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b_size)
    });

    Ok(((tch_a, tch_b), (a, b)))
}

#[allow(unused)]
fn common_input_i64(
    lhs_shape: &[i64],
    rhs_shape: &[i64],
) -> anyhow::Result<((TchTensor, TchTensor), (Tensor<i64>, Tensor<i64>))> {
    let tch_a = TchTensor::arange(
        lhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(lhs_shape);
    let mut a = Tensor::<i64>::empty(lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const i64, a_size)
    });

    let tch_b = TchTensor::arange(
        rhs_shape.iter().product::<i64>(),
        (tch::Kind::Int64, tch::Device::Cpu),
    )
    .reshape(rhs_shape);
    let mut b = Tensor::<i64>::empty(rhs_shape)?;
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
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let len = rng.random_range(1..=3);
        let shape = (1..=len)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape)?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b);
        assert_method(&c, &tch_c);
    }
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
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let len = rng.random_range(2..=4);
        let mut shape = (1..=len)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let mut shape2 = shape.clone();
        shape2[0] = 1;
        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape2)?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b);
        assert_method(&c, &tch_c);
        shape[1] = 1;
        shape2[0] = 1;
        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape2)?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b);
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = vec![13, 13, 13];

        let start = rng.random_range(0..5);
        let end = rng.random_range((start + 1)..10);
        let step = rng.random_range(1..=2);

        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape)?;
        let tch_a = tch_a
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let a = slice!(a[start:end:step,start:end:step,start:end:step])?;
        let tch_b = tch_b
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let b = slice!(b[start:end:step,start:end:step,start:end:step])?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b);
        assert_method(&c, &tch_c);
    }
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
)]
#[test]
fn fn_name() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = (1..=3)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape)?;
        let tch_a = tch_a.permute(&[2, 1, 0][..]);
        let a = a.permute([2, 1, 0])?;
        let tch_b = tch_b.permute(&[2, 1, 0][..]);
        let b = b.permute([2, 1, 0])?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b).contiguous();
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = vec![13, 13, 13];

        let start = rng.random_range(0..5);
        let end = rng.random_range((start + 1)..10);
        let step = rng.random_range(1..=2);
        let ((tch_a, tch_b), (a, b)) = input_method(&shape, &shape)?;
        let tch_a = tch_a
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let a = slice!(a[start:end:step,start:end:step,start:end:step])?;
        let tch_b = tch_b
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let b = slice!(b[start:end:step,start:end:step,start:end:step])?;
        let tch_a = tch_a.permute(&[2, 1, 0][..]);
        let a = a.permute([2, 1, 0])?;
        let tch_b = tch_b.permute(&[2, 1, 0][..]);
        let b = b.permute([2, 1, 0])?;
        let c = hpt_op;
        let tch_c = tch_a.tch_op(&tch_b).contiguous();
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = (1..=3)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape)?;
        let c = a.clone().hpt_op(scalar);
        let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let len = rng.random_range(2..=4);
        let mut shape = (1..=len)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let mut shape2 = shape.clone();
        shape2[0] = 1;
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape2)?;
        let c = a.clone().add(scalar);
        let tch_c = tch_a.shallow_clone().add(&TchTensor::from(scalar));
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
        assert_method(&c, &tch_c);
        shape[1] = 1;
        shape2[0] = 1;
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape2)?;
        let c = a.clone().hpt_op(scalar);
        let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = vec![13, 13, 13];

        let start = rng.random_range(0..5);
        let end = rng.random_range((start + 1)..10);
        let step = rng.random_range(1..=2);
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape)?;
        let tch_a = tch_a
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let a = slice!(a[start:end:step,start:end:step,start:end:step])?;
        let c = a.clone().hpt_op(scalar);
        let tch_c = tch_a.shallow_clone().tch_op(&TchTensor::from(scalar));
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a);
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = (1..=3)
            .map(|_| rng.random_range(1..=10))
            .collect::<Vec<_>>();
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape)?;
        let tch_a = tch_a.permute(&[2, 1, 0][..]);
        let a = a.permute([2, 1, 0])?;
        let c = a.clone().hpt_op(scalar);
        let tch_c = tch_a
            .shallow_clone()
            .tch_op(&TchTensor::from(scalar))
            .contiguous();
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a).contiguous();
        assert_method(&c, &tch_c);
    }
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
    let mut rng = rand::rng();
    for _ in 0..100 {
        let shape = vec![13, 13, 13];

        let start = rng.random_range(0..5);
        let end = rng.random_range((start + 1)..10);
        let step = rng.random_range(1..=2);
        let ((tch_a, _), (a, _)) = input_method(&shape, &shape)?;
        let tch_a = tch_a
            .slice(0, start, end, step)
            .slice(1, start, end, step)
            .slice(2, start, end, step);
        let a = slice!(a[start:end:step,start:end:step,start:end:step])?;
        let tch_a = tch_a.permute(&[2, 1, 0][..]);
        let a = a.permute([2, 1, 0])?;
        let c = a.clone().hpt_op(scalar);
        let tch_c = tch_a
            .shallow_clone()
            .tch_op(&TchTensor::from(scalar))
            .contiguous();
        assert_method(&c, &tch_c);
        let c = scalar.hpt_op(&a);
        let tch_c = TchTensor::from(scalar).tch_op(&tch_a).contiguous();
        assert_method(&c, &tch_c);
    }
    Ok(())
}

#[test]
fn test_batch_matmul() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input(&[13, 13, 13], &[13, 13, 13])?;
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
