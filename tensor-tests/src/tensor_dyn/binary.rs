#![allow(unused_imports)]
use rayon::iter::{ IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator };
use tch::Tensor;
use tensor_common::slice;
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::TensorInfo;
use tensor_dyn::ShapeManipulate;
use std::ops::*;
use tensor_dyn::Random;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;
use tensor_dyn::slice::SliceOps;

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
                a_raw[i],
                b_raw[i],
                abs_diff,
                relative_diff
            );
        }
    }
}

fn common_input<const N: usize, const M: usize>(
    lhs_shape: [i64; N],
    rhs_shape: [i64; M]
) -> anyhow::Result<((Tensor, Tensor), (_Tensor<f64>, _Tensor<f64>))> {
    let tch_a = Tensor::randn(&lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let a = _Tensor::<f64>::empty(&lhs_shape)?;
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a.size())
    });

    let tch_b = Tensor::randn(&rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let b = _Tensor::<f64>::empty(&rhs_shape)?;
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b.size())
    });

    Ok(((tch_a, tch_b), (a, b)))
}

#[test]
fn test_add() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 10])?;
    let c = a.add(&b);
    let tch_c = tch_a.add(&tch_b);
    assert_eq(&c, &tch_c);
    Ok(())
}

#[test]
fn test_add_broadcast() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 1])?;
    let c = a.add(&b);
    let tch_c = tch_a.add(&tch_b);
    assert_eq(&c, &tch_c);

    let ((tch_a, tch_b), (a, b)) = common_input([1, 10], [10, 1])?;
    let c = a.add(&b);
    let tch_c = tch_a.add(&tch_b);
    assert_eq(&c, &tch_c);
    Ok(())
}

#[test]
fn test_add_sub_tensors() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 10])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1, 2:6:1])?;
    let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let b = slice!(b[2:6:1, 2:6:1])?;
    let c = a.add(&b);
    let tch_c = tch_a.add(&tch_b);
    assert_eq(&c, &tch_c);
    Ok(())
}
