// use env_logger::{ Builder, Env };
// use hpt_dyn::set_num_threads;
// use hpt_dyn::{ onnx::load_onnx, DType, Device, Tensor as DynTensor };
// use log::LevelFilter;
// use std::collections::HashMap;
// use hpt::common::{ Pointer, TensorInfo };
// use hpt::ops::*;
// use hpt::Tensor;
use tract_onnx::prelude::*;
fn main() -> anyhow::Result<()> {
    // let model = load_onnx("resnet34.onnx").expect("加载模型失败");
    // let mut map = HashMap::new();
    // map.insert("input".to_string(), DynTensor::ones(&[1, 3, 768, 768], DType::F32, Device::Cpu)?);
    // let initialized = model.initialize()?;

    // let now = std::time::Instant::now();
    // for _ in 0..100 {
    //     let res = initialized.execute(&map)?;
    // }
    // println!("Time taken: {:?}", now.elapsed() / 100);

    for i in 0..20 {
        let mut model = tract_onnx::onnx().model_for_path("resnet34.onnx")?;
        // 设置输入维度
        let model = model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 160 + i * 32, 160 + i * 32))
            )?
            .into_optimized()?
            .into_runnable()?;

        let input = tract_ndarray::Array4::<f32>
            ::ones([1, 3, 160 + i * 32, 160 + i * 32])
            .into_dyn()
            .into_arc_tensor();

        let now = std::time::Instant::now();
        for _ in 0..100 {
            let _outputs = model.run(tvec!(tract_onnx::prelude::TValue::Const(input.clone())))?;
        }
        println!("Time taken: {:?}", now.elapsed() / 100);
    }
    Ok(())
}
