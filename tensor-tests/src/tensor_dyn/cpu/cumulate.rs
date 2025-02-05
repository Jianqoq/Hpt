#![allow(unused_imports)]

use std::i64;

use duplicate::duplicate_item;
use rand::Rng;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{RandomInt, Tensor, TensorCreator};
use tensor_macros::match_selection;

#[allow(unused)]
fn assert_eq(b: &Tensor<i64>, a: &tch::Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            panic!(
                "{} != {}, bytes: {:?}, {:?}",
                a_raw[i],
                b_raw[i],
                a_raw[i].to_ne_bytes(),
                b_raw[i].to_ne_bytes()
            );
        }
    }
}

#[duplicate_item(
    func            hpt_method      tch_method;
    [test_cumsum]   [cumsum]        [cumsum];
    [test_cumprod]  [cumprod]       [cumprod];
)]
#[test]
fn func() -> anyhow::Result<()> {
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        let len = rng.gen_range(1..3);
        let mut shape = Vec::with_capacity(len);
        for _ in 0..len {
            shape.push(rng.gen_range(1..10));
        }
        let tch_a = tch::Tensor::randint_low(
            i64::MIN,
            i64::MAX,
            &shape,
            (tch::Kind::Int64, tch::Device::Cpu),
        );
        let mut a = Tensor::<i64>::empty(shape)?;
        a.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_a.data_ptr() as *mut i64, tch_a.numel())
        });

        let dim = rng.gen_range(0..len) as i64;
        let b = a.hpt_method(dim)?;
        let tch_b = tch_a.tch_method(dim, tch::Kind::Int64);
        assert_eq(&b, &tch_b);
    }
    Ok(())
}
