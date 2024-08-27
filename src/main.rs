use std::hint::black_box;
use half::bf16;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0i64, 8 * 2048 * 2048)?.reshape(&[8, 2048, 2048])?;
    // let b = _Tensor::<f32>::randn(&[104, 104, 104, 104])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let _ = a.sin()?;
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(8 * 2048 * 2048, (Kind::Float, Device::Cpu)).reshape(&[8, 2048, 2048]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let res = a.sin();
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
