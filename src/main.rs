use std::hint::black_box;
use std::io::Write;

use ops::cpu::conv_config::{ Conv2dConfig, KernelParamAlgo };
use rust_xlsxwriter::{ Format, Workbook };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

struct KernelCombination {
    kk: usize,
    v: usize,
    i: usize,
    l: usize,
    m: usize,
    n: usize,
    j: usize,
    k: usize,
    ii: usize,
    b: usize,
}

impl KernelCombination {
    fn new(kk: usize, v: usize, i: usize, l: usize, m: usize, n: usize, j: usize, k: usize, ii: usize, b: usize) -> Self {
        KernelCombination { kk, v, i, l, m, n, j, k, ii, b }
    }

    // 计算寄存器重用
    fn compute_register_reuse(&self) -> usize {
        // 假设寄存器的重用次数为 i * m * n，对于其他部分你可以根据你的情况调整
        self.i * self.m * self.n
    }

    // 计算缓存占用
    fn compute_cache_usage(&self, cache_size: usize, l1_cache_size: usize, l2_cache_size: usize) -> usize {
        let data_size = self.kk * self.v * self.i * self.l * self.m * self.n * self.j * self.k * self.ii * self.b;
        
        // 如果缓存不足，就增加缓存溢出的成本
        if data_size <= l1_cache_size {
            1  // 如果数据大小在L1缓存中能完全容纳
        } else if data_size <= l2_cache_size {
            10 // 如果数据大小在L2缓存中能完全容纳
        } else {
            100 // 超出L2缓存，访问主存
        }
    }

    // 计算总成本
    fn compute_total_cost(&self, l1_cache_size: usize, l2_cache_size: usize) -> usize {
        let register_reuse_cost = self.compute_register_reuse();
        let cache_usage_cost = self.compute_cache_usage(10000, l1_cache_size, l2_cache_size);
        
        // 假设寄存器重用和缓存使用的权重各占一半
        register_reuse_cost + cache_usage_cost
    }
}

fn evaluate_combinations() {
    let combinations = vec![
        KernelCombination::new(5, 2, 2, 3, 3, 3, 128, 50, 512, 1),
        KernelCombination::new(5, 2, 8, 3, 3, 3, 128, 50, 512, 1),
    ];

    let l1_cache_size = 10000; // 假设L1缓存能容纳10000个f32
    let l2_cache_size = 320000; // 假设L2缓存能容纳320000个f32

    for (idx, comb) in combinations.iter().enumerate() {
        let cost = comb.compute_total_cost(l1_cache_size, l2_cache_size);
        println!("组合 {} 的总成本是: {}", idx + 1, cost);
    }
}


fn main() -> anyhow::Result<()> {
    // evaluate_combinations();
    set_num_threads(16);
    let oc = 128;
    let ic = 4096*4;
    let kh = 3;
    let kw = 3;
    let h = 256;
    let w = 256;
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
    let config = Conv2dConfig::<f32>::new(oc, ic, [kh, kw], KernelParamAlgo::Greedy);
    let now = std::time::Instant::now();
    let res = black_box(a.iconv2d(
        &kernel,
        [1, 1],
        [
            (0, 0),
            (0, 0),
        ],
        [1, 1],
        Some(&config)
    )?);
    println!("{:?}", now.elapsed());
    // println!("{:?}", res);
    // conv2d()?;

    Ok(())
}

fn conv2d() -> Result<(), anyhow::Error> {
    let ic_sets = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let oc_sets = [128];
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
