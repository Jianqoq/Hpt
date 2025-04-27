use std::time::Instant;

use hpt::ops::Matmul;
use hpt::types::f16;
use hpt::{ops::*, Tensor};
use hpt_dyn::Device;
use hpt_dyn::{DType, Tensor as DynTensor};
fn main() -> anyhow::Result<()> {
    let a = DynTensor::ones(&[512, 512], DType::F32, Device::Cpu)?;
    let start = Instant::now();
    for _ in 0..100 {
        let c = a.sigmoid()?;
        // println!("c: {}", c);
    }
    println!("time: {:?}", start.elapsed() / 100);
    let a = Tensor::<f32>::ones(&[512, 512])?;
    let b = Tensor::<f32>::randn(&[512, 2048])?;
    let start = Instant::now();
    for _ in 0..100 {
        let c = a.sigmoid()?;
    }
    println!("time: {:?}", start.elapsed() / 100);

    // let start = Instant::now();
    // for _ in 0..10000 {
    //     let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    // }
    // println!("time: {:?}", start.elapsed() / 10000);
    Ok(())
}
