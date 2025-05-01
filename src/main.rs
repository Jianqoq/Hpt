use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor};
use std::collections::HashMap;
fn main() -> anyhow::Result<()> {
    let model = load_onnx("model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        Tensor::randn(0.0, 1.0, &[1, 3, 384, 384], DType::F16, Device::Cpu)?,
    );
    model.initialize()?.execute(map).expect("执行模型失败");
    // println!("{:#?}", model);

    Ok(())
}
