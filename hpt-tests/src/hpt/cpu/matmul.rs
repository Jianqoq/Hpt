#![allow(unused)]
use hpt::buitin_templates::cpu::gemm;
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::Conv;
use hpt::ops::Matmul;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::Tensor;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOut;
use hpt_types::type_promote::NormalOutUnary;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tch;

use super::assert_utils::assert_f64;

#[test]
fn test() -> anyhow::Result<()> {
    let mut rng = rand::rng();
    for i in 0..1000 {
        let m = rng.random_range(1..=512 * 3);
        let n = rng.random_range(1..=512 * 3);
        let k = rng.random_range(1..=512 * 3);
        println!("{} {} {}", m, n, k);
        let a = Tensor::<f32>::arange(0, m * k)?.reshape(&[m, k])?;
        let b = Tensor::<f32>::arange(0, k * n)?.reshape(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = gemm(&a, &b, None, 16)?;
        assert!(c.allclose(&c2));
    }
    Ok(())
}
