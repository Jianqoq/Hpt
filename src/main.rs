use std::ffi::CStr;
use std::hint::black_box;

use half::bf16;
use ops::cpu::convolutions::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn assert_eq(a: &Tensor, b: &_Tensor<i64>) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();
    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            println!("{} != {}", a_raw[i], b_raw[i]);
        }
    }
}
fn main() -> anyhow::Result<()> {
    let a = black_box(_Tensor::<f32>::randn([64, 64, 64]).unwrap());
    let a1 = black_box(_Tensor::<f32>::randn([64, 64, 64]).unwrap());
    let a2 = black_box(_Tensor::<f32>::randn([64, 64, 64]).unwrap());
    for i in 0..1 {
        _Tensor::<f32>::stack(vec![&a, &a1, &a2], 0 as usize, false).unwrap();
    }
    println!("{:?}", a2);
    // set_global_display_precision(7);
    // set_global_display_lr_elements(6);
    // set_num_threads(16);
    // let kernel = _Tensor::<f32>
    //     ::arange(0, 80 * 80 * 4 * 4)?
    //     .reshape([80, 80, 4, 4])?
    //     .permute([2, 3, 1, 0])?
    //     .contiguous()?;
    // let a = _Tensor::<f32>
    //     ::arange(0, 1 * 80 * 1260 * 1260)?
    //     .reshape([1, 80, 1260, 1260])?
    //     .permute([0, 2, 3, 1])?
    //     .contiguous()?;
    // let config = Conv2dConfig::<f32>::new(
    //     80,
    //     80,
    //     [4, 4],
    //     KernelParamAlgo::Greedy
    // );
    // println!("config: {:?}", config);
    // let now = std::time::Instant::now();
    // for _ in 0..100 {
    //     let res = a.conv2d(
    //         &kernel,
    //         [1, 1],
    //         [
    //             (0, 0),
    //             (0, 0),
    //         ],
    //         [1, 1],
    //         Some(&config)
    //     )?;
    // }
    // println!("{:?}", now.elapsed() / 100);

    Ok(())
}
