use env_logger::{Builder, Env};
use log::LevelFilter;
use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // Builder::new()
    //     .filter_level(LevelFilter::Debug
    //     )
    //     .init();
    let model = load_onnx("model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        DynTensor::ones(&[1, 3, 224, 224], DType::F32, Device::Cpu)?,
    );
    let initialized = model.initialize()?;

    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = initialized.execute(10, map.clone())?;
        println!("res: {:#?}", res);
        // println!("next");
    }
    let duration = now.elapsed();
    println!("Time taken: {:?}", duration / 100);
    Ok(())
}
