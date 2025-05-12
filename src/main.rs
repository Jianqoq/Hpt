use env_logger::{Builder, Env};
use hpt_dyn::{onnx::load_onnx, DType, Device, Tensor as DynTensor};
use log::LevelFilter;
use std::collections::HashMap;
use hpt::common::{Pointer, TensorInfo};
use hpt::ops::*;
use hpt::Tensor;

fn main() -> anyhow::Result<()> {
    // Builder::new().filter_level(LevelFilter::Off).init();
    // let model = load_onnx("lstm.onnx").expect("加载模型失败");
    // let mut map = HashMap::new();
    // map.insert(
    //     "input".to_string(),
    //     DynTensor::ones(&[1, 512, 128], DType::F32, Device::Cpu)?,
    // );
    // let initialized = model.initialize()?;

    // let now = std::time::Instant::now();
    // for _ in 0..10 {
    //     let res = initialized.execute(16, map.clone())?;
    //     // println!("res: {}", res["output"]);
    //     // println!("next");
    // }
    // println!("Time taken: {:?}", now.elapsed() / 10);

    // let dyn_a = DynTensor::ones(&[1, 224, 224, 3], DType::F32, Device::Cpu)?;
    // let dyn_b = DynTensor::ones(&[1, 224, 224, 3], DType::F32, Device::Cpu)?;

    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let res = &dyn_a + &dyn_b;
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);

    // let hpt_a = Tensor::<f32>::ones([1, 224, 224, 3])?;
    // let hpt_b = Tensor::<f32>::ones([1, 224, 224, 3])?;

    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let res = &hpt_a + &hpt_b;
    // }
    // let duration = now.elapsed();
    // println!("Time taken: {:?}", duration / 10000);


    fn new_matmul(
        a: Pointer<f32>, // [m, k]
        b: Pointer<f32>, // [k, n]
        mut out: Pointer<f32>, // [m, n]
        m: usize,
        n: usize,
        k: usize,
        nc: usize,
        mc: usize,
        kc: usize,
    ) {
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += a[mm * kc + kk] * b[kk * nc + nn];
                }
                out[mm * n + nn] = acc;
            }
        }
    }

    let hpt_a = Tensor::<f32>::ones([64, 128])?;
    let hpt_b = Tensor::<f32>::ones([128, 256])?;
    let c = Tensor::<f32>::ones([64, 256])?;

    let now = std::time::Instant::now();
    for _ in 0..100 {
        new_matmul(
            hpt_a.ptr(),
            hpt_b.ptr(),
            c.ptr(),
            64,
            256,
            128,
            0, 0, 0,
        );
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
