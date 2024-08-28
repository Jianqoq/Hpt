use std::hint::black_box;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<f32>::randn(&[256, 256, 256])?;
    let b = _Tensor::<f32>::arange(0i64, 256 * 256)?.reshape(&[256])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let res = a.selu(None, None)?;
    });
    println!("hpt time: {:?}", now.elapsed() /100);

    let a = Tensor::randn([256, 256, 256], (Kind::Float, Device::Cpu));
    let b = Tensor::arange(1 * 256, (Kind::Float, Device::Cpu)).reshape(&[256]);

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let res: Tensor = a.selu();
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
