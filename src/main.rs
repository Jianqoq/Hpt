use std::time::Instant;

// use hpt::ops::Matmul;
// use hpt::utils::set_num_threads;
use hpt::{ops::*, Tensor};
use hpt_dyn::{DType, Device, Tensor as DynTensor};
fn main() -> anyhow::Result<()> {
    // set_num_threads(10);
    let a = DynTensor::ones(&[512, 512], DType::F32, Device::Cpu)?;
    let start = Instant::now();

    for _ in 0..10000 {
        let c = a.sin()?;
    }
    println!("time: {:?}", start.elapsed() / 10000);

    let a = Tensor::<f32>::ones(&[512, 512])?;
    let start = Instant::now();
    for _ in 0..10000 {
        let c = a.sin()?;
    }
    println!("time: {:?}", start.elapsed() / 10000);

    Ok(())
}
