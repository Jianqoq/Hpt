use std::hint::black_box;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(3);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::arange(0, 256 * 256 * 2024)?.reshape(&[256, 256, 2024])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = a.sum([0, 1], false)?;
        println!("{:?}", res);
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(256 * 256 * 2024, (Kind::Float, Device::Cpu)).reshape(&[256, 256, 2024]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let res = a.sum_dim_intlist(&[0, 1][..], false, Kind::Float);
        println!("{}", res);
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
