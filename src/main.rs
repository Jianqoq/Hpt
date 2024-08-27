use std::hint::black_box;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0i64, 8 * 64 * 64 * 64 * 256)?.reshape(&[8, 64, 64, 64, 256])?;
    let b = _Tensor::<f32>::arange(0i64, 8 * 64 * 64 * 64 * 256)?.reshape(&[8, 64, 64, 64, 256])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = a.sin()?;
    });
    println!("hpt time: {:?}", now.elapsed() /100);

    let a = Tensor::arange(8 * 64 * 64 * 64 * 256, (Kind::Float, Device::Cpu)).reshape(&[8, 64, 64, 64, 256]);
    let b = Tensor::arange(8 * 64 * 64 * 64 * 256, (Kind::Float, Device::Cpu)).reshape(&[8, 64, 64, 64, 256]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = a.sin();
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
