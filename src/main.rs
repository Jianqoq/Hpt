use env_logger::{Builder, Env};
use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use log::LevelFilter;
use std::collections::HashMap;
use hpt::common::{Pointer, TensorInfo};
use hpt::ops::*;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    // Builder::new().filter_level(LevelFilter::Off).init();
    let model = load_onnx("lstm.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        DynTensor::ones(&[128, 256, 20], DType::F32, Device::Cpu)?,
    );
    let initialized = model.initialize()?;
    
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = initialized.execute(&map)?;
        // println!("res: {}", res["output"]);
        // println!("next");
    }
    println!("Time taken: {:?}", now.elapsed() / 100);

    // let m = 512;
    // let n = 512;
    // let k = 512;
    // let dyn_a = DynTensor::randn(0.0, 1.0, &[m, k], DType::F16, Device::Cpu)?;
    // let dyn_b = DynTensor::randn(0.0, 1.0, &[k, n], DType::F16, Device::Cpu)?;
    // println!("dyn_a: {}", dyn_a);
    // println!("dyn_b: {}", dyn_b);
    // let dyn_a_f32 = dyn_a.cast(DType::F32)?;
    // let dyn_b_f32 = dyn_b.cast(DType::F32)?;
    // let bias = DynTensor::ones(&[n], DType::F32, Device::Cpu)?;

    // let now = std::time::Instant::now();
    // for _ in 0..1 {
    //     let res = dyn_a.matmul(&dyn_b)?;
    //     let res_f32 = dyn_a_f32.matmul(&dyn_b_f32)?;
    //     println!("{}", res);
    //     println!("{}", res_f32);
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

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
