// use std::ffi::CStr;
// use std::hint::black_box;

// use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tch::{ Device, Kind, Tensor };
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
    set_num_threads(16);
    let kernel = _Tensor::<half::f16>
        ::arange(0, 16 * 16 * 3 * 3)?
        .reshape([16, 16, 3, 3])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<half::f16>
        ::arange(0, 16 * 16 * 512 * 512)?
        .reshape([16, 16, 512, 512])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<half::f16>::new(16, 16, [3, 3], KernelParamAlgo::Greedy);
    println!("config: {:?}", config);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let res = a
            .conv2d_ex(
                &kernel,
                [5, 5],
                [
                    (0, 0),
                    (0, 0),
                ],
                [5, 5],
                Some(&config)
            )?
            .permute([0, 3, 1, 2])?;
        // println!("{:?}", res);
    }
    println!("{:?}", now.elapsed() / 100);
    // // println!("{:?}", res);
    // let tch_a = Tensor::arange(16 * 16 * 512 * 512, (Kind::Int64, Device::Cpu)).reshape(
    //     &[16, 16, 512, 512]
    // );
    // let tch_kernel = Tensor::arange(16 * 16 * 3 * 3, (Kind::Int64, Device::Cpu)).reshape(
    //     &[16, 16, 3, 3]
    // );
    // let res = tch_a.conv2d(&tch_kernel, None::<Tensor>, &[5, 5], &[0, 0], &[5, 5], 1);
    // println!("{}", res);
    Ok(())
}
