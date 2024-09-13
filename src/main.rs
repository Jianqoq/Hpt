// use std::ffi::CStr;
// use std::hint::black_box;

// use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
// use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

// fn assert_eq(a: &Tensor, b: &_Tensor<i64>) {
//     let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
//     let b_raw = b.as_raw();
//     for i in 0..b.size() {
//         if a_raw[i] != b_raw[i] {
//             println!("{} != {}", a_raw[i], b_raw[i]);
//         }
//     }
// }
fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    let kernel = _Tensor::<f32>
        ::arange(0, 16 * 1024 * 8 * 8)?
        .reshape([16, 1024, 8, 8])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 16 * 1024 * 256 * 256)?
        .reshape([16, 1024, 256, 256])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(16, 1024, [8, 8], KernelParamAlgo::Greedy);
    println!("config: {:?}", config);
    let now = std::time::Instant::now();
    for _ in 0..2 {
        let res = a
            .conv2d_ex(
                &kernel,
                [1, 1],
                [
                    (0, 0),
                    (0, 0),
                ],
                [1, 1],
                Some(&config)
            )?
            .permute([0, 3, 1, 2])?;
        // println!("{:?}", res);
    }
    println!("{:?}", now.elapsed() / 2);
    // println!("{:?}", res);
    Ok(())
}
