use hpt::Tensor;
use hpt_dyn::set_num_threads;
use hpt_dyn::{ onnx::load_onnx, DType, Device, Tensor as DynTensor };
use std::collections::HashMap;
use hpt::common::TensorInfo;
use hpt::ops::*;
// use tract_onnx::prelude::*;
fn main() -> anyhow::Result<()> {
    // let mut res = Vec::new();
    // for i in 9..10 {
    //     let model = load_onnx("resnet34.onnx").expect("加载模型失败");
    //     let mut map = HashMap::new();
    //     map.insert("input".to_string(), DynTensor::ones(&[1, 3, 64 + i * 32, 64 + i * 32], DType::F16, Device::Cpu)?);
    //     let initialized = model.initialize()?;

    //     let now = std::time::Instant::now();
    //     for _ in 0..50 {
    //         let res = initialized.execute(&map)?;
    //         // println!("res: {    }", res["output"]);
    //     }
    //     res.push(now.elapsed() / 50);
    // }
    // println!("Time taken: {:?}", res);

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

    let m = 29;
    let n = 4;
    let k = 2;
    println!("m: {}, n: {}, k: {}", m, n, k);
    let a = Tensor::<f32>::randn(&[m, k])?;
    let b = Tensor::<f32>::randn(&[k, n])?;
    println!("a: {}", a);
    println!("b: {}", b);
    let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
        println!("c: {}", c);
    let a3 = (unsafe {
        DynTensor::from_raw(
            a.ptr().ptr as *mut u8,
            a.layout().clone(),
            hpt_dyn::DType::F32,
            hpt_dyn::Device::Cpu,
            false
        )
    })?;
    let b3 = (unsafe {
        DynTensor::from_raw(
            b.ptr().ptr as *mut u8,
            b.layout().clone(),
            hpt_dyn::DType::F32,
            hpt_dyn::Device::Cpu,
            false
        )
    })?;
    let c3 = a3.matmul(&b3)?;
    // assert!(c.allclose(&c2, TEST_ATOL, TEST_RTOL));
    let c3_hpt: Tensor<f32> = (unsafe { Tensor::from_raw(c3.ptr().ptr as *mut f32, c3.shape()) })?;
    println!("c3: {}", c3_hpt);
    Ok(())
}
