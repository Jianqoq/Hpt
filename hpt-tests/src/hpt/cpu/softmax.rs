#![allow(unused)]
use duplicate::duplicate_item;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt_common::slice;
use rand::Rng;
use tch::Tensor;

use crate::{TestTypes, EPSILON, TCH_TEST_TYPES, TEST_ATOL, TEST_RTOL};

use super::assert_utils::assert_f64;

#[allow(unused)]
#[track_caller]
fn assert_eq_f64(hpt_res: &hpt::Tensor<TestTypes>, tch_res: &Tensor) {
    let tch_res = unsafe {
        hpt::Tensor::<TestTypes>::from_raw(tch_res.data_ptr() as *mut TestTypes, &hpt_res.shape().to_vec())
    }.expect("from_raw failed");
    assert!(hpt_res.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
}

fn common_input(end: i64, shape: &[i64]) -> anyhow::Result<(hpt::Tensor<TestTypes>, Tensor)> {
    let a = hpt::Tensor::<TestTypes>::arange(0, end)?.reshape(shape)?;
    let tch_a = Tensor::arange(end, (TCH_TEST_TYPES, tch::Device::Cpu)).reshape(shape);
    Ok((a, tch_a))
}

#[duplicate_item(
    func                    hpt_method      tch_method;
    [test_softmax]          [softmax]       [softmax];
    [test_logsoftmax]       [log_softmax]   [log_softmax];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..100 {
        let len = rng.random_range(1..3);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.random_range(1..10));
        }
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        let dim = rng.random_range(0..len) as i64;
        let res = a.hpt_method(dim)?;
        let tch_res = tch_a.tch_method(dim, TCH_TEST_TYPES);
        assert_eq_f64(&res, &tch_res);
    }
    Ok(())
}

#[test]
fn test_layernorm() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for _ in 0..1000 {
        let len = rng.random_range(1..3);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.random_range(1..10));
        }
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        let to_normalize_dims = rng.random_range(1..=len);
        let mut shape = a.shape().iter().rev();
        let mut normalized_shape = vec![0; to_normalize_dims];
        for i in (0..to_normalize_dims).rev() {
            normalized_shape[i] = *shape.next().unwrap();
        }
        let (gamma, tch_gamma) =
            common_input(normalized_shape.iter().product::<i64>(), &normalized_shape)?;
        let (beta, tch_beta) =
            common_input(normalized_shape.iter().product::<i64>(), &normalized_shape)?;
        let res = a.layernorm(&normalized_shape, Some(&gamma), Some(&beta), EPSILON)?;
        let tch_res = tch_a.layer_norm(
            &normalized_shape,
            Some(tch_gamma),
            Some(tch_beta),
            1e-5,
            false,
        );
        assert_eq_f64(&res, &tch_res);
    }
    Ok(())
}
