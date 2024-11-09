use rust_xlsxwriter::{ Format, Workbook };
use std::io::Write;
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;
use tensor_dyn::type_promote::NormalOut;
use tensor_dyn::type_promote::FloatOutUnary;
use tensor_dyn::type_promote::NormalOutUnary;

fuse_proc_macro!(
fn feedforward(a: _Tensor<f32>, b: _Tensor<f32>) -> anyhow::Result<_Tensor<f32>> {
    let c = &a + &b;
    let d = c.sin()?;
    let e = d.relu()?;
    let f = &a * &b;
    Ok(f)
});

#[fuse]
fn feedforward2(a: _Tensor<f32>, b: _Tensor<f32>) -> anyhow::Result<_Tensor<f32>> {
    let c = &a + &b;
    let d = c.sin()?;
    let e = d.relu()?;
    let f = &a * &b;
    Ok(f)
}

fn main() -> anyhow::Result<()> {
    conv2d()?;
    Ok(())
}

fn conv2d() -> Result<(), anyhow::Error> {
    let oc_sets = [128];
    let ic_sets = [4096];
    let kh_sets = [3];
    let kw_sets = [3];
    let h_sets = [256];
    let w_sets = [256];

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
                                .reshape([ic, oc, kh, kw])?
                                .permute([2, 3, 1, 0])?
                                .contiguous()?;
                            let a = _Tensor::<f32>
                                ::arange(0, 1 * ic * h * w)?
                                .reshape([1, ic, h, w])?
                                .permute([0, 2, 3, 1])?
                                .contiguous()?;
                            // let device = Device::Cpu;
                            // let a = Tensor::randn(1.0, 1.0, &[1, ic, h, w], &device)?;
                            // let kernel = Tensor::randn(1.0, 1.0, &[oc, ic, kh, kw], &device)?;
                            let now = std::time::Instant::now();
                            let _ = a.conv2d(
                                &kernel,
                                None,
                                [1, 1],
                                [
                                    (0, 0),
                                    (0, 0),
                                ],
                                [1, 1],
                                None
                            )?;
                            // println!("{:?}", res.shape());
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
