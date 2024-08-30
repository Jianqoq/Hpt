use std::hint::black_box;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;


fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0, 8 * 8096 * 2048)?.reshape(&[8, 8096, 2048])?;
    // println!("{:?}", a);
    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = _Tensor::<f32>::hamming_window(8 * 8096 * 2048, true)?;
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(8 * 8096 * 2048, (Kind::Float, Device::Cpu)).reshape(&[8, 8096, 2048]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = Tensor::hamming_window_periodic(8 * 8096 * 2048, true, (Kind::Float, Device::Cpu));
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
