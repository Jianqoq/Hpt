use std::time::Instant;

use hpt::ops::Matmul;
use hpt::{ops::*, Tensor};
// use hpt_dyn::Tensor;
fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::ones(&[128, 512])?;
    let b = Tensor::<f32>::ones(&[512, 512])?;
    let start = Instant::now();

    for _ in 0..100000 {
        let c = a.matmul(&b).unwrap();
    }
    println!("time: {:?}", start.elapsed() / 100000);

    Ok(())
}
