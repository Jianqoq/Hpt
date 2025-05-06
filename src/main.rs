use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use std::collections::HashMap;

use hpt::ops::*;
use hpt::Tensor;
fn main() -> anyhow::Result<()> {
    let model = load_onnx("lstm_model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        DynTensor::ones(&[1, 1, 20], DType::F32, Device::Cpu)?,
    );
    let initialized = model.initialize()?;

    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = initialized.execute(10, map.clone())?;
        println!("res: {}", res["output"]);
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 100);

    // let a = DynTensor::randn(0.0, 1.0, &[1, 256], DType::F32, Device::Cpu)?;
    // let b = DynTensor::randn(0.0, 1.0, &[256, 256], DType::F32, Device::Cpu)?;
    // let bias = DynTensor::randn(0.0, 1.0, &[256], DType::F32, Device::Cpu)?;
    // let now = std::time::Instant::now();
    // for _ in 0..1 {
    //     let c1 = a.addmm(&b, &bias)?;
    //     println!("c: {}", c1);
    //     let c2 = a.matmul(&b)? + bias.clone();
    //     println!("c: {}", c2);
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

    // let a = Tensor::<f32>::ones(&[1, 256])?;
    // let b = Tensor::<f32>::ones(&[256, 4 * 256])?;
    // let now = std::time::Instant::now();
    // spindle::with_lock(10, || {
    //     for _ in 0..10000 {
    //         let c = a.matmul(&b).unwrap();
    //         // println!("c: {}", c);
    //     }
    // });
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

    Ok(())
}
