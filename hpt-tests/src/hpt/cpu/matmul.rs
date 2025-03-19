#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::Conv;
use hpt::ops::Random;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::ops::{Gemm, Matmul};
use hpt::slice;
use hpt::utils::resize_cpu_lru_cache;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;
use tch::Tensor as TchTensor;

use super::assert_utils::assert_f64;

#[allow(unused)]
#[allow(unused)]
fn common_input<const N: usize, const M: usize>(
    lhs_shape: [i64; N],
    rhs_shape: [i64; M],
) -> anyhow::Result<((TchTensor, TchTensor), (hpt::Tensor<f64>, hpt::Tensor<f64>))> {
    let tch_a = TchTensor::randn(&lhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut a = hpt::Tensor::<f64>::empty(&lhs_shape)?;
    let a_size = a.size();
    a.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_a.data_ptr() as *const f64, a_size)
    });

    let tch_b = TchTensor::randn(&rhs_shape, (tch::Kind::Double, tch::Device::Cpu));
    let mut b = hpt::Tensor::<f64>::empty(&rhs_shape)?;
    let b_size = b.size();
    b.as_raw_mut().copy_from_slice(unsafe {
        std::slice::from_raw_parts(tch_b.data_ptr() as *const f64, b_size)
    });

    Ok(((tch_a, tch_b), (a, b)))
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

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_matmul_sub_tensors() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 10])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let b = slice!(b[2:6:1,2:6:1])?;
    let empty = Tensor::<f64>::empty(&[4, 4])?;
    let c = a.matmul_(&b, empty)?;
    let tch_c = tch_a.matmul(&tch_b);
    assert_eq_10(&c, &tch_c);
    Ok(())
}
#[test]
fn test_matmul_uncontiguous() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 10])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let tch_b = tch_b.permute(&[1, 0][..]);
    let b = b.permute([1, 0])?;
    let mut empty = Tensor::<f64>::empty(&[10, 10])?;
    let c = a.matmul_(&b, &mut empty)?;
    let tch_c = tch_a.matmul(&tch_b).contiguous();
    assert_eq_10(&c, &tch_c);
    Ok(())
}
#[test]
fn test_matmul_uncontiguous_sub_tensors() -> anyhow::Result<()> {
    let ((tch_a, tch_b), (a, b)) = common_input([10, 10], [10, 10])?;
    let tch_a = tch_a.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let a = slice!(a[2:6:1,2:6:1])?;
    let tch_b = tch_b.slice(0, 2, 6, 1).slice(1, 2, 6, 1);
    let b = slice!(b[2:6:1,2:6:1])?;
    let tch_a = tch_a.permute(&[1, 0][..]);
    let a = a.permute([1, 0])?;
    let tch_b = tch_b.permute(&[1, 0][..]);
    let b = b.permute([1, 0])?;
    let mut empty = Tensor::<f64>::empty(&[4, 4])?;
    let c = a.matmul_(&b, &mut empty)?;
    let tch_c = tch_a.matmul(&tch_b).contiguous();
    assert_eq_10(&c, &tch_c);
    Ok(())
}
