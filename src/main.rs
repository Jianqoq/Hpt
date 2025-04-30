use hpt_dyn::onnx::load_onnx;
use std::collections::HashMap;
fn main() -> anyhow::Result<()> {
    let model = load_onnx("model.onnx").expect("加载模型失败");
    model.execute(HashMap::new()).expect("执行模型失败");
    // println!("{:#?}", model);

    Ok(())
}
