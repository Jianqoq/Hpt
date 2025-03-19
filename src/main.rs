use std::arch::aarch64::vld1q_s16_x4;

use hpt::types::f16;
use hpt::{
    ops::{Gemm, Matmul, TensorCreator},
    utils::resize_cpu_lru_cache,
    Tensor,
};
fn main() -> anyhow::Result<()> {
    let m = 512 * 4;
    let n = 512 * 4;
    let k = 512 * 4;
    let a = Tensor::<f16>::ones(&[m, k])?;
    let b = Tensor::<f16>::ones(&[k, n])?;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let c = a.matmul(&b)?;
    }
    println!("{:?}", now.elapsed() / 100);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let c2 = a.gemm(
            &b,
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            false,
            false,
            false,
        )?;
    }
    println!("{:?}", now.elapsed() / 100);

    Ok(())
}
