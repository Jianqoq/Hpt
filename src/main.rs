use std::hint::black_box;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0, 8 * 1 * 2048)?.reshape(&[8, 1, 2048])?;
    let b = _Tensor::<f32>::arange(0, 8 * 8096 * 1)?.reshape(&[8, 8096, 1])?;
    // println!("{:?}", a);
    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        // let _ = a.mean([1], false);
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(8 * 1 * 2048, (Kind::Float, Device::Cpu)).reshape(&[8, 1, 2048]);
    let b = Tensor::arange(8 * 8096 * 1, (Kind::Float, Device::Cpu)).reshape(&[8, 8096, 1]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let _ = &a + &b;
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
