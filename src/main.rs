use std::hint::black_box;
use half::bf16;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0i64, 1 * 2048)?.reshape(&[1, 2048])?;
    let b = _Tensor::<f32>::arange(0i64, 1 * 2048)?.reshape(&[2048, 1])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = &a + &b;
        println!("{:?}", res);
    });
    println!("hpt time: {:?}", now.elapsed() / 10);

    let a = Tensor::arange(1 * 2048, (Kind::Float, Device::Cpu)).reshape(&[1, 2048]);
    let b = Tensor::arange(1 * 2048, (Kind::Float, Device::Cpu)).reshape(&[2048, 1]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = &a + &b;
    });
    println!("torch time: {:?}", now.elapsed() / 10);
    Ok(())
}
