use env_logger::{ Builder, Env };
use log::LevelFilter;
use hpt_dyn::{ onnx::load_onnx, DType, Device, Tensor as DynTensor };
use std::collections::HashMap;

use hpt::ops::*;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    // Builder::new().filter_level(LevelFilter::Off).init();
    let model = load_onnx("model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert("input".to_string(), DynTensor::ones(&[1, 3, 384, 384], DType::F32, Device::Cpu)?);
    let initialized = model.initialize()?;

    let now = std::time::Instant::now();
    for _ in 0..10 {
        let res = initialized.execute(16, map.clone())?;
        // println!("res: {}", res["output"]);
        // println!("next");
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 10);

    // let dyn_a = DynTensor::ones(&[1, 224, 224, 3], DType::F32, Device::Cpu)?;
    // let dyn_b = DynTensor::ones(&[1, 224, 224, 3], DType::F32, Device::Cpu)?;

    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let res = &dyn_a + &dyn_b;
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
