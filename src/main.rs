use hpt::common::cpu::TensorLike;
use hpt::common::TensorInfo;
use hpt::ops::Random;
use hpt::types::{bf16, f16, Complex32};
use hpt::utils::resize_cpu_lru_cache;
use hpt::{
    ops::{Gemm, Matmul, TensorCreator},
    Tensor,
};
use rand::Rng;
fn main() -> anyhow::Result<()> {
    /*  m: 237, n: 1, k: 226
    m: 418, n: 102, k: 2
    m: 75, n: 1, k: 228
    m: 438, n: 42, k: 2 */
    resize_cpu_lru_cache(1, 0);
    let mut rng = rand::rng();
    for i in 0..1 {
        let m = 438 * 1;
        let n = 42 * 1;
        let k = 2 * 1;
        println!("{i}: {} {} {}", m, n, k);
        let a = Tensor::<f32>::randn(&[m, k])?;
        let b = Tensor::<f32>::randn(&[k, n])?;
        let now = std::time::Instant::now();
        // for _ in 0..100 {
        let c = a.matmul(&b)?;
        // }
        // println!("matmul: {:?}", now.elapsed() / 100);
        let now = std::time::Instant::now();
        // for _ in 0..100 {
        let c2 = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        // }
        // println!("gemm: {:?}", now.elapsed() / 100);
        println!("c: {}", c);
        println!("c2: {}", c2);
        println!("allclose: {}", c.allclose(&c2, 1.0e-5, 1.0e-5));
    }
    Ok(())
}
