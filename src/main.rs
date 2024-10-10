use std::io::Write;

use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use rust_xlsxwriter::{ Format, Workbook };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    // set_num_threads(16);
    // let oc = 1024;
    // let ic = 1024 * 4;
    // let kh = 3;
    // let kw = 3;
    // let h = 256;
    // let w = 256;
    // let kernel = _Tensor::<f32>
    //     ::arange(0, oc * ic * kh * kw)?
    //     .reshape([oc, ic, kh, kw])?
    //     // .permute([0, 2, 3, 1])?
    //     .permute([2, 3, 1, 0])?
    //     .contiguous()?;
    // // let kernel = _Tensor::<f32>::randn([kh, kw, ic, oc])?;
    // let a = _Tensor::<f32>
    //     ::arange(0, 1 * ic * h * w)?
    //     .reshape([1, ic, h, w])?
    //     .permute([0, 2, 3, 1])?
    //     .contiguous()?;
    // // let a = _Tensor::<f32>::randn([1, h, w, ic])?;
    // let config = Conv2dConfig::<f32>::new(oc, ic, [kh, kw], KernelParamAlgo::Greedy);
    // let now = std::time::Instant::now();
    // for _ in 0..1 {
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
    //     // let res2 = a.conv2d(
    //     //     &kernel,
    //     //     [1, 1],
    //     //     [
    //     //         (0, 0),
    //     //         (0, 0),
    //     //     ],
    //     //     [1, 1],
    //     //     Some(&config)
    //     // )?;
    //     // assert_eq!(res, res2);
    // }
    // println!("{:?}", now.elapsed() / 1);
    conv2d()?;

    Ok(())
}

fn conv2d() -> Result<(), anyhow::Error> {
    let oc_sets = [1024, 2048];
    let ic_sets = [1024, 2048];
    let kh_sets = [3];
    let kw_sets = [3];
    let h_sets = [256, 512];
    let w_sets = [256, 512];

    set_num_threads(16);
    let mut workbook = Workbook::new();
    let decimal_format = Format::new().set_num_format("0.0000000000");
    let format = Format::new();
    let worksheet = workbook.add_worksheet();

    let mut row = 0;
    for ic in ic_sets {
        for oc in oc_sets {
            for kh in kh_sets {
                for kw in kw_sets {
                    for h in h_sets {
                        for w in w_sets {
                            let kernel = _Tensor::<f32>
                                ::arange(0, oc * ic * kh * kw)?
                                .reshape([oc, ic, kh, kw])?
                                // .permute([0, 2, 3, 1])?
                                .permute([2, 3, 1, 0])?
                                .contiguous()?;
                            let a = _Tensor::<f32>
                                ::arange(0, 1 * ic * h * w)?
                                .reshape([1, ic, h, w])?
                                .permute([0, 2, 3, 1])?
                                .contiguous()?;
                            let config = Conv2dConfig::<f32>::new(
                                oc,
                                ic,
                                [kh, kw],
                                KernelParamAlgo::Greedy
                            );
                            let now = std::time::Instant::now();
                            let _ = a.iconv2d(
                                &kernel,
                                [1, 1],
                                [
                                    (0, 0),
                                    (0, 0),
                                ],
                                [1, 1],
                                Some(&config)
                            )?;
                            worksheet.write_number(
                                row,
                                0,
                                now.elapsed().as_micros() as f64,
                                &decimal_format
                            )?;
                            worksheet.write_string(
                                row,
                                1,
                                &format!("({}, {}, {}, {}, {}, {})", ic, oc, kh, kw, h, w),
                                &format
                            )?;
                            print!(
                                "\rprogress: {}%",
                                ((row + 1) * 100) /
                                    (
                                        (ic_sets.len() *
                                            oc_sets.len() *
                                            kh_sets.len() *
                                            kw_sets.len() *
                                            h_sets.len() *
                                            w_sets.len()) as u32
                                    )
                            );
                            std::io::stdout().flush().expect("Failed to flush stdout");
                            row += 1;
                        }
                    }
                }
            }
        }
    }

    workbook.save("conv2d_result.xlsx")?;
    Ok(())
}
