use env_logger::{Builder, Env};
use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use log::LevelFilter;
use std::collections::HashMap;
use hpt::common::{Pointer, TensorInfo};
use hpt::ops::*;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    // Builder::new().filter_level(LevelFilter::Off).init();
    // let model = load_onnx("lstm.onnx").expect("加载模型失败");
    // let mut map = HashMap::new();
    // map.insert(
    //     "input".to_string(),
    //     DynTensor::ones(&[128, 256, 20], DType::F32, Device::Cpu)?,
    // );
    // let initialized = model.initialize()?;
    
    // let now = std::time::Instant::now();
    // for _ in 0..100 {
    //     let res = initialized.execute(&map)?;
    //     // println!("res: {}", res["output"]);
    //     // println!("next");
    // }
    // println!("Time taken: {:?}", now.elapsed() / 100);

    let m = 8;
    let n = 512;
    let k = 512;
    let dyn_a = DynTensor::randn(0.0, 1.0, &[m, k], DType::F32, Device::Cpu)?;
    let dyn_b = DynTensor::randn(0.0, 1.0, &[k, n], DType::F32, Device::Cpu)?;

    let hpt_a = unsafe { Tensor::<f32>::from_raw(dyn_a.ptr().ptr as *mut f32, &[m, k]) }?;
    let hpt_b = unsafe { Tensor::<f32>::from_raw(dyn_b.ptr().ptr as *mut f32, &[k, n]) }?;
    let now = std::time::Instant::now();
    for _ in 0..10000 {
        let res1 = dyn_a.matmul(&dyn_b)?;
        // let res2 = hpt_a.gemm(&hpt_b, 0.0, 1.0, false, false, false)?;
        // println!("res1: {}", res1);
        // println!("res2: {}", res2);
    }
    
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 10000);

    // let hpt_a = Tensor::<f32>::ones([1, 224, 224, 3])?;
    // let hpt_b = Tensor::<f32>::ones([1, 224, 224, 3])?;

    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let res = &hpt_a + &hpt_b;
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

    Ok(())
}
