use hpt::types::{bf16, f16, Complex32};
use hpt::utils::resize_cpu_lru_cache;
use hpt::{
    ops::{Gemm, Matmul, TensorCreator},
    Tensor,
};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    resize_cpu_lru_cache(1, 0);
    let mut rng = rand::rng();
    for i in 0..5000 {
        let m = rng.random_range(1..=512 * 4);
        let n = rng.random_range(1..=512 * 4);
        let k = rng.random_range(1..=512 * 4);
        println!("{i}: {} {} {}", m, n, k);
        let a = Tensor::<Complex32>::ones(&[m, k])?;
        let b = Tensor::<Complex32>::ones(&[k, n])?;
        let c = a.matmul(&b)?;
        let c2 = a.gemm(&b, 0.0.into(), 1.0.into(), false, false, false)?;
        assert!(c.allclose(&c2));
    }
    Ok(())
}
