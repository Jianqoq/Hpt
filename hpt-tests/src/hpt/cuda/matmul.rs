#![allow(unused)]
use super::assert_utils::assert_f64;
use hpt::backend::Cpu;
use hpt::backend::Cuda;
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
use hpt::utils::set_seed;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;
use tch::Tensor as TchTensor;

#[allow(unused)]
fn assert_eq_10(b: &Tensor<f32, Cuda>, a: &Tensor<f32, Cpu>) {
    let b_cpu = b.to_cpu::<0>().expect("failed to convert to cpu");
    let a_raw = a.as_raw();
    let b_raw = b_cpu.as_raw();

    for i in 0..b.size() {
        let abs_diff = (a_raw[i] - b_raw[i]).abs();
        let rel_diff = if a_raw[i] == 0.0 && b_raw[i] == 0.0 {
            0.0
        } else {
            abs_diff / (a_raw[i].abs() + b_raw[i].abs() + f32::EPSILON)
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
        let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        let a_cuda = a.to_cuda::<0>()?;
        let b_cuda = b.to_cuda::<0>()?;
        let c_cuda = a_cuda.matmul(&b_cuda)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
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
        let a_cuda = a.to_cuda::<0>()?;
        let b_cuda = b.to_cuda::<0>()?;
        let c_cuda = a_cuda.matmul(&b_cuda)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_t() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[n, k])?;
        let c = a.gemm(&b.t()?, 0.0, 1.0, false, false, false)?;
        let a_cuda = a.to_cuda::<0>()?;
        let b_cuda = b.to_cuda::<0>()?;
        let c_cuda = a_cuda.matmul(&b_cuda.t()?)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
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
        let a_cuda = a.to_cuda::<0>()?;
        let b_cuda = b.to_cuda::<0>()?;
        let c_cuda = a_cuda.matmul(&b_cuda.t()?)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
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
        let a = Tensor::<f32>::randn(&[k, m])?;
        let b = Tensor::<f32>::randn(&[n, k])?;
        let c = a.t()?.gemm(&b.t()?, 0.0, 1.0, false, false, false)?;
        let a_cuda = a.to_cuda::<0>()?.t()?;
        let b_cuda = b.to_cuda::<0>()?.t()?;
        let c_cuda = a_cuda.matmul(&b_cuda)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
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
        let a_cuda = a.to_cuda::<0>()?.t()?;
        let b_cuda = b.to_cuda::<0>()?.t()?;
        let c_cuda = a_cuda.matmul(&b_cuda)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
    }
    Ok(())
}

#[test]
fn test_batch_matmul() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..100 {
        let m = rng.random_range(1..=512);
        let n = rng.random_range(1..=512);
        let k = rng.random_range(1..=512);
        let dim0 = rng.random_range(1..=2);
        let dim1 = rng.random_range(1..=2);
        let a = Tensor::<f32>::randn(&[dim0, dim1, m, k])?;
        let b = Tensor::<f32>::randn(&[dim0, dim1, k, n])?;
        let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        let a_cuda = a.to_cuda::<0>()?;
        let b_cuda = b.to_cuda::<0>()?;
        let c_cuda = a_cuda.matmul(&b_cuda)?;
        assert!(c.allclose(&c_cuda.to_cpu::<0>()?, 1.0e-3, 1.0e-3));
    }
    Ok(())
}
