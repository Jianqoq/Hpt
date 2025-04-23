use std::time::Instant;

use hpt::ops::Matmul;
use hpt::{ops::*, Tensor};
fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::randn(&[128, 512])?;
    let b = Tensor::<f32>::randn(&[512, 2048])?;
    let start = Instant::now();
    for _ in 0..1000 {
        let c = a.matmul(&b)?;
    }
    println!("time: {:?}", start.elapsed() / 1000);

    // let start = Instant::now();
    // for _ in 0..10000 {
    //     let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    // }
    // println!("time: {:?}", start.elapsed() / 10000);
    Ok(())
}
