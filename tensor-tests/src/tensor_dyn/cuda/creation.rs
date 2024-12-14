#![allow(unused_imports)]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_dyn::Cuda;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorInfo;
use tensor_dyn::TensorLike;
use tensor_dyn::{Tensor, TensorCreator};
use tensor_macros::match_selection;

#[test]
fn test_arange() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::arange(0, 100)?;
    Ok(())
}

#[test]
fn test_zeros() -> anyhow::Result<()> {
    let tch_a = tch::Tensor::zeros(&[1000], (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64>::zeros(&[1000])?;
    Ok(())
}

#[test]
fn test_full() -> anyhow::Result<()> {
    // let tch_a = tch::Tensor::full(&[1000], 1.0, (tch::Kind::Double, tch::Device::Cpu));
    let a = Tensor::<f64, Cuda, 0>::full(1.0, &[1000])?;
    // assert_eq(&a, &tch_a);
    Ok(())
}
