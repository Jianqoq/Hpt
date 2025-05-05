#![allow(unused)]
use crate::TestTypes;
use crate::EPSILON;
use crate::TEST_ATOL;
use crate::TEST_RTOL;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::*;
use hpt::slice;
use hpt::types::TypeCommon;
use hpt::utils::resize_cpu_lru_cache;
use hpt::Tensor;
// use hpt_dyn::Tensor as DynTensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use hpt_types::type_promote::NormalOutUnary2;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;
use tch::Tensor as TchTensor;

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..1000 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        // let a3 = unsafe { DynTensor::from_raw(
        //     a.ptr().ptr as *mut u8,
        //     a.layout().clone(),
        //     hpt_dyn::DType::F32,
        //     hpt_dyn::Device::Cpu,
        //     false,
        // ) }?;
        // let b3 = unsafe { DynTensor::from_raw(
        //     b.ptr().ptr as *mut u8,
        //     b.layout().clone(),
        //     hpt_dyn::DType::F32,
        //     hpt_dyn::Device::Cpu,
        //     false,
        // ) }?;
        // let c3 = a3.matmul(&b3)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
        // let c3_hpt: Tensor<f32> = unsafe { Tensor::from_raw(c3.ptr().ptr as *mut TestTypes, c3.shape()) }?;
        // assert!(c.allclose(&c3_hpt, TEST_ATOL, TEST_RTOL));
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
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[k, n])?;
        let c = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        let c2 = a.matmul(&b)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_post_op() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[k, n])?;
        let c = a.matmul_post(&b, |x, _, _| x._relu(), |x, _, _| x._relu())?;
        let c2 = a
            .gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?
            .relu()?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_mp_post_op() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[k, n])?;
        let c = a.matmul_post(&b, |x, _, _| x._relu(), |x, _, _| x._relu())?;
        let c2: Tensor<TestTypes> = a.matmul(&b)?.relu()?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_t() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[n, k])?.t()?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
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
        let a = Tensor::<TestTypes>::randn(&[m, k])?;
        let b = Tensor::<TestTypes>::randn(&[n, k])?;
        let c = a.gemm(
            &b.t()?,
            TestTypes::ZERO,
            TestTypes::ONE,
            false,
            false,
            false,
        )?;
        let c2 = a.matmul(&b.t()?)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_t_t() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let a = Tensor::<TestTypes>::randn(&[k, m])?.t()?;
        let b = Tensor::<TestTypes>::randn(&[n, k])?.t()?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
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
        let a = Tensor::<TestTypes>::randn(&[k, m])?;
        let b = Tensor::<TestTypes>::randn(&[n, k])?;
        let c = a.t()?.gemm(
            &b.t()?,
            TestTypes::ZERO,
            TestTypes::ONE,
            false,
            false,
            false,
        )?;
        let c2 = a.t()?.matmul(&b.t()?)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_batch_matmul() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let dim0 = rng.random_range(1..=2);
        let dim1 = rng.random_range(1..=2);
        let a = Tensor::<TestTypes>::randn(&[dim0, dim1, m, k])?;
        let b = Tensor::<TestTypes>::randn(&[dim0, dim1, k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}

#[test]
fn test_uncontiguous_batch_matmul() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..10 {
        let m = rng.random_range(1..=128);
        let n = rng.random_range(1..=128);
        let k = rng.random_range(1..=128);
        let dim0 = rng.random_range(1..=4);
        let dim1 = rng.random_range(1..=4);
        let a = Tensor::<TestTypes>::randn(&[dim0, dim1, k, m])?.permute(&[1, 0, 3, 2])?;
        let b = Tensor::<TestTypes>::randn(&[dim0, dim1, n, k])?.permute(&[1, 0, 3, 2])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, TestTypes::ZERO, TestTypes::ONE, false, false, false)?;
        assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    }
    Ok(())
}
