use std::time::Instant;

use hpt::ops::Matmul;
use hpt::utils::set_num_threads;
use hpt::{ops::*, Tensor};
fn main() -> anyhow::Result<()> {
    set_num_threads(16);
    let a = Tensor::<f32>::randn(&[128, 512])?;
    let b = Tensor::<f32>::randn(&[512, 2048])?;
    let start = Instant::now();
    for _ in 0..1000 {
        let c = a.matmul(&b)?;
    }
    println!("time: {:?}", start.elapsed() / 1000);
    Ok(())
}
