use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let model = load_onnx("lstm_model.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        Tensor::randn(0.0, 1.0, &[128, 512, 256], DType::F32, Device::Cpu)?,
    );
    model.initialize()?.execute(map)?;

    // let a = Tensor::from_vec(vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9], &[3, 3])?;
    // println!("{}", a);

    // let b = Tensor::from_vec(vec![0i64, 2], &[1, 2])?;
    // println!("{}", b);

    // let c = a.gather(&b, 1)?;
    // println!("{}", c);

    Ok(())
}
