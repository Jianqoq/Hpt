use std::hint::black_box;

use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

const IN: i64 = 512;
const OUT: i64 = 128;
const KH: i64 = 3;
const KW: i64 = 3;
const H: i64 = 256;
const W: i64 = 256;

fn main() -> anyhow::Result<()> {
    // black_box(mem_access_pattern_test());

    conv2d()?;

    Ok(())
}

fn conv2d() -> Result<(), anyhow::Error> {
    set_num_threads(16);
    let kernel = _Tensor::<f32>
        ::arange(0, OUT * IN * KH * KW)?
        .reshape([OUT, IN, KH, KW])?
        // .permute([0, 2, 3, 1])?
        .permute([2, 3, 1, 0])?
        .contiguous()?;
    let a = _Tensor::<f32>
        ::arange(0, 1 * IN * H * W)?
        .reshape([1, IN, H, W])?
        .permute([0, 2, 3, 1])?
        .contiguous()?;
    let config = Conv2dConfig::<f32>::new(OUT, IN, [KH, KW], KernelParamAlgo::Greedy);
    let now = std::time::Instant::now();
    for _ in 0..5 {
        let res = a.iconv2d(
            &kernel,
            [1, 1],
            [
                (0, 0),
                (0, 0),
            ],
            [1, 1],
            Some(&config)
        )?;
        // println!("{:?}", res);
    }
    let elapsed0 = now.elapsed() / 5;
    println!("{:?}", elapsed0);

    // let kernel = _Tensor::<f32>
    //     ::arange(0, OUT * IN * 2 * KH * KW)?
    //     .reshape([OUT, IN * 2, KH, KW])?
    //     // .permute([0, 2, 3, 1])?
    //     .permute([2, 3, 1, 0])?
    //     .contiguous()?;
    // let a = _Tensor::<f32>
    //     ::arange(0, 1 * IN * 2 * H * W)?
    //     .reshape([1, IN * 2, H, W])?
    //     .permute([0, 2, 3, 1])?
    //     .contiguous()?;
    // let config = Conv2dConfig::<f32>::new(OUT, IN, [KH, KW], KernelParamAlgo::Greedy);
    // let now = std::time::Instant::now();
    // for _ in 0..5 {
    //     let res = a.iconv2d(
    //         &kernel,
    //         [1, 1],
    //         [
    //             (0, 0),
    //             (0, 0),
    //         ],
    //         [1, 1],
    //         Some(&config)
    //     )?;
    //     // println!("{:?}", res);
    // }
    // let elapsed1 = now.elapsed() / 5;

    // let kernel = _Tensor::<f32>
    //     ::arange(0, OUT * IN * 2 * 2 * KH * KW)?
    //     .reshape([OUT, IN * 2 * 2, KH, KW])?
    //     // .permute([0, 2, 3, 1])?
    //     .permute([2, 3, 1, 0])?
    //     .contiguous()?;
    // let a = _Tensor::<f32>
    //     ::arange(0, 1 * IN * 2 * 2 * H * W)?
    //     .reshape([1, IN * 2 * 2, H, W])?
    //     .permute([0, 2, 3, 1])?
    //     .contiguous()?;
    // let config = Conv2dConfig::<f32>::new(OUT, IN, [KH, KW], KernelParamAlgo::Greedy);
    // let now = std::time::Instant::now();
    // for _ in 0..5 {
    //     let res = a.iconv2d(
    //         &kernel,
    //         [1, 1],
    //         [
    //             (0, 0),
    //             (0, 0),
    //         ],
    //         [1, 1],
    //         Some(&config)
    //     )?;
    //     // println!("{:?}", res);
    // }
    // let elapsed2 = now.elapsed() / 5;

    // println!("expected x 2: {:?}, got: {:?}, expected x 4: {:?}, got: {:?}", elapsed0 * 2, elapsed1, elapsed0 * 4, elapsed2);
    Ok(())
}

fn mem_access_pattern_test() {
    use std::time::Instant;
    // 创建一个大矩阵
    let rows = 10000;
    let cols = 16000;
    let mut matrix: Vec<Vec<f32>> = vec![vec![0.0; cols]; rows];

    // 计时按顺序读取
    let start_seq = Instant::now();
    black_box(for i in 0..rows {
        let row_ptr = matrix[i].as_mut_ptr(); // 获取行的裸指针
        for j in 0..cols {
            unsafe {
                let value = row_ptr.add(j); // 访问每个元素
                // 读取数据，实际操作
                *value += 1.0;
            }
        }
    });
    let duration_seq = start_seq.elapsed();
    println!("顺序读取时间: {:?}", duration_seq);

    // let vec_size = 16;
    // let l_end = cols / vec_size;
    // // // 计时跳行读取
    // let start_jump = Instant::now();
    // black_box(for l in 0..l_end {
    //     for i in 0..rows {
    //         let row_ptr = matrix[i].as_mut_ptr(); // 获取行的裸指针
    //         for j in 0..vec_size {
    //             let j = l * vec_size + j;
    //             unsafe {
    //                 let value = row_ptr.add(j); // 访问每个元素
    //                 // 读取数据，实际操作
    //                 *value += 1.0;
    //             }
    //         }
    //     }
    // });
    // let duration_jump = start_jump.elapsed();
    // println!("跳行读取时间: {:?}", duration_jump);
}
