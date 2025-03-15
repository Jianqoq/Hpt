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
    let mut rng = rand::thread_rng();
    for i in 0..100 {
        let m = rng.gen_range(1..=512);
        let n = rng.gen_range(1..=512);
        let k = rng.gen_range(1..=512);
        let a = Tensor::<f32>::arange(0, m * k)?.reshape(&[m, k])?;
        let b = Tensor::<f32>::arange(0, k * n)?.reshape(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = gemm(&a, &b, None)?;
        assert!(c.allclose(&c2));
    }
    Ok(())
}
