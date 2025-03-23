#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::slice;
use hpt::utils::resize_cpu_lru_cache;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::type_promote::NormalOutUnary2;
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
fn test_special_shape() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let shapes = [
        [1, 1, 1],
        [1, 1, 128],
        [1, 128, 1],
        [1, 128, 128],
        [128, 1, 1],
        [128, 1, 128],
        [128, 128, 1],
        [128, 128, 128],
    ];
    for [m, n, k] in shapes {
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[k, n])?;
        let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        let c2 = a.matmul(&b)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_post_op() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[k, n])?;
        let c = a.matmul_post(&b, |x| x._relu(), |x| x._relu())?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?.relu()?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_mp_post_op() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<half::bf16>::randn(&[m, k])?;
        let b = Tensor::<half::bf16>::randn(&[k, n])?;
        let c = a.matmul_post(&b, |x| x._relu(), |x| x._relu())?;
        let c2: Tensor<half::bf16> = a.matmul(&b)?.relu()?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_t() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..2 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[n, k])?.t()?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_t_special_shape() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let shapes = [
        [1, 1, 1],
        [1, 1, 128],
        [1, 128, 1],
        [1, 128, 128],
        [128, 1, 1],
        [128, 1, 128],
        [128, 128, 1],
        [128, 128, 128],
    ];
    for [m, n, k] in shapes {
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[n, k])?;
        let c = a.gemm(&b.t()?, 0.0, 1.0, false, false, false)?;
        let c2 = a.matmul(&b.t()?)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_t_t() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<f32>::randn(&[k, m])?.t()?;
        let b = Tensor::<f32>::randn(&[n, k])?.t()?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_t_t_special_shape() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    let shapes = [
        [1, 1, 1],
        [1, 1, 128],
        [1, 128, 1],
        [1, 128, 128],
        [128, 1, 1],
        [128, 1, 128],
        [128, 128, 1],
        [128, 128, 128],
    ];
    for [m, n, k] in shapes {
        let a = Tensor::<f32>::randn(&[k, m])?;
        let b = Tensor::<f32>::randn(&[n, k])?;
        let c = a.t()?.gemm(&b.t()?, 0.0, 1.0, false, false, false)?;
        let c2 = a.t()?.matmul(&b.t()?)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_batch_matmul() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let dim0 = rng.random_range(1..=2);
        let dim1 = rng.random_range(1..=2);
        let a = Tensor::<f32>::randn(&[dim0, dim1, m, k])?;
        let b = Tensor::<f32>::randn(&[dim0, dim1, k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_uncontiguous_batch_matmul() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let dim0 = rng.random_range(1..=4);
        let dim1 = rng.random_range(1..=4);
        let a = Tensor::<f32>::randn(&[dim0, dim1, k, m])?.permute(&[1, 0, 3, 2])?;
        let b = Tensor::<f32>::randn(&[dim0, dim1, n, k])?.permute(&[1, 0, 3, 2])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2, 1.0e-3, 1.0e-3));
    }
    Ok(())
}
