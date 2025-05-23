use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use std::collections::HashMap;
// use tract_onnx::prelude::*;
fn main() -> anyhow::Result<()> {
    let mut res = Vec::new();
    let model = load_onnx("lstm.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        DynTensor::ones(&[128, 128, 512], DType::F32, Device::Cpu)?,
        // DynTensor::ones(&[128, 128], DType::F32, Device::Cpu)?,
    );
    let initialized = model.initialize()?;
    let now = std::time::Instant::now();
    // spindle::with_lock(8, || {
    for _ in 0..100 {
        let res = initialized.execute(&map).unwrap();
        // println!("res: {}", res["output"]);
    }
    // });
    res.push(now.elapsed() / 100);
    // }
    println!("Time taken: {:?}", res);
    Ok(())
}
