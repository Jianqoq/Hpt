use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor};
use std::collections::HashMap;

// use hpt::Tensor;
// use hpt::ops::*;
fn main() -> anyhow::Result<()> {
    let model = load_onnx("lstm_model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        Tensor::randn(0.0, 1.0, &[128, 512, 256], DType::F32, Device::Cpu)?,
    );
    let initialized = model.initialize()?;

    let now = std::time::Instant::now();
    for _ in 0..1 {
        initialized.execute(map.clone())?;
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 1);

    // let a = Tensor::randn(0.0, 1.0, &[128, 512, 256], DType::F32, Device::Cpu)?;
    // let b = Tensor::randn(0.0, 1.0, &[256, 20], DType::F32, Device::Cpu)?;
    // let now = std::time::Instant::now();
    // for _ in 0..100 {
    //     let c = a.matmul(&b)?;
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 100);

    //     let a = Tensor::<f32>::randn(&[512, 256])?;
    // let b = Tensor::<f32>::randn(&[256, 20])?;
    // let now = std::time::Instant::now();
    // for _ in 0..1 {
    //     let c = a.matmul(&b)?;
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

    Ok(())
}
