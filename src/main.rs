use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use std::collections::HashMap;
// use tract_onnx::prelude::*;
fn main() -> anyhow::Result<()> {
    let mut res = Vec::new();
    // for i in 0..1 {
    let model = load_onnx("lstm.onnx").expect("加载模型失败");
    let mut map = HashMap::new();
    map.insert(
        "input".to_string(),
        DynTensor::ones(&[128, 256, 128], DType::F32, Device::Cpu)?,
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

    // for i in 0..20 {
    //     let mut model = tract_onnx::onnx().model_for_path("resnet34.onnx")?;
    //     // 设置输入维度
    //     let model = model
    //         .with_input_fact(
    //             0,
    //             InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 160 + i * 32, 160 + i * 32))
    //         )?
    //         .into_optimized()?
    //         .into_runnable()?;

    //     let input = tract_ndarray::Array4::<f32>
    //         ::ones([1, 3, 160 + i * 32, 160 + i * 32])
    //         .into_dyn()
    //         .into_arc_tensor();

    //     let now = std::time::Instant::now();
    //     for _ in 0..100 {
    //         let _outputs = model.run(tvec!(tract_onnx::prelude::TValue::Const(input.clone())))?;
    //     }
    //     println!("Time taken: {:?}", now.elapsed() / 100);
    // }
    Ok(())
}
