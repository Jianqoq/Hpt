#![allow(unused)]
use duplicate::duplicate_item;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt_common::slice;
use rand::Rng;
use tch::Tensor;

use crate::{TestTypes, TCH_TEST_TYPES, TEST_ATOL, TEST_RTOL};

use super::assert_utils::assert_f64;

#[allow(unused)]
#[track_caller]
fn assert_eq(hpt_res: &hpt::Tensor<TestTypes>, tch_res: &Tensor) {
    let tch_res = unsafe {
        hpt::Tensor::<TestTypes>::from_raw(
            tch_res.data_ptr() as *mut TestTypes,
            &hpt_res.shape().to_vec(),
        )
    }
    .expect("from_raw failed");
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
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let len = 2;
    let shapes = [[1, 13], [2, 1024], [3, 1123], [3, 4096], [3, 5551]];
    for shape in shapes {
        let (a, tch_a) = common_input(shape.iter().product::<i64>(), &shape)?;

        for dim in 0..len {
            let res = a.to_cuda::<0>()?.hpt_method(dim)?;
            let tch_res = tch_a.tch_method(dim, TCH_TEST_TYPES);
            assert_eq(&res.to_cpu::<0>()?, &tch_res);
        }
    }
    Ok(())
}
