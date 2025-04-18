use std::time::Instant;

use hpt::{ops::Random, Tensor};
use hpt::ops::Matmul;
fn main() -> anyhow::Result<()> {
    let n = 1024;
    let a = Tensor::<f32>::randn(&[32, 512])?;
    let b = Tensor::<f32>::randn(&[512, 512])?;
    let start = Instant::now();
    for _ in 0..100000 {
        let c = a.matmul(&b)?;
    }
    println!("time: {:?}", start.elapsed() / 100000);

    Ok(())
}
