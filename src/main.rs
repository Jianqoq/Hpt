use std::time::Instant;

use hpt::ops::Matmul;
use hpt::types::f16;
use hpt::{ops::*, Tensor};
use hpt_dyn::Device;
use hpt_dyn::{DType, Tensor as DynTensor};
fn main() -> anyhow::Result<()> {
    // let a = DynTensor::ones(&[512, 512, 512], DType::F32, Device::Cpu)?;
    // let start = Instant::now();
    // for _ in 0..100 {
    //     let c = a.sum(&[0], false)?;
    //     // println!("c: {}", c);
    // }
    // println!("time: {:?}", start.elapsed() / 100);
    // let a = Tensor::<f32>::ones(&[512, 512, 512])?;
    // let start = Instant::now();
    // for _ in 0..100 {
    //     let c = a.sum(&[0], false)?;
    // }
    // println!("time: {:?}", start.elapsed() / 100);

    let x = DynTensor::randn(0.0, 1.0, &[2, 3, 4], DType::F32, Device::Cpu)?;
    let mut out = DynTensor::empty(&[2, 4], DType::F32, Device::Cpu)?;
    let y = x.sum_(&[1], true, &mut out)?;
    Ok(())
}
