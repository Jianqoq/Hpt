#![allow(unused)]
use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Contiguous;
use hpt::ops::Conv;
use hpt::ops::ShapeManipulate;
use hpt::ops::TensorCreator;
use hpt::ops::{Gemm, Matmul};
use hpt::utils::resize_cpu_lru_cache;
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
    resize_cpu_lru_cache(1, 0);
    let mut rng = rand::rng();
    for i in 0..5000 {
        let m = rng.random_range(1..=512 * 4);
        let n = rng.random_range(1..=512 * 4);
        let k = rng.random_range(1..=512 * 4);
        println!("{i}: {} {} {}", m, n, k);
        let a = Tensor::<f32>::ones(&[m, k])?;
        let b = Tensor::<f32>::ones(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        assert!(c.allclose(&c2));
    }
    Ok(())
}
