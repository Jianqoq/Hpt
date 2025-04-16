#![allow(unused_imports)]

use std::i64;

use duplicate::duplicate_item;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::CumulativeOps;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_common::slice;
use rand::Rng;

use crate::TestTypes;
use crate::TCH_TEST_TYPES;
use crate::TEST_ATOL;
use crate::TEST_RTOL;

#[duplicate_item(
    func            hpt_method      tch_method;
    [test_cumsum]   [cumsum]        [cumsum];
    [test_cumprod]  [cumprod]       [cumprod];
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
        let tch_a = tch::Tensor::randn(&shape, (TCH_TEST_TYPES, tch::Device::Cpu));
        let mut a = Tensor::<TestTypes>::empty(shape)?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *mut TestTypes, tch_a.numel())
        });

        let dim = rng.random_range(0..len) as i64;
        let b = a.hpt_method(dim)?;
        let tch_b = tch_a.tch_method(dim, TCH_TEST_TYPES);
        let tch_res = unsafe {
            Tensor::<TestTypes>::from_raw(tch_b.data_ptr() as *mut TestTypes, &b.shape().to_vec())
        }?;
        assert!(b.allclose(&tch_res, TEST_RTOL, TEST_ATOL));
    }
    Ok(())
}
