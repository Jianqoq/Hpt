use std::hint::black_box;
use half::bf16;
use tch::{ Device, Kind, Tensor };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(6);
    set_num_threads(10);
    tch::set_num_threads(10);
    let a = _Tensor::<bf16>::arange(0i64, 8 * 2048 * 2048)?.reshape(&[8, 2048, 2048])?;
    // let b = _Tensor::<f32>::randn(&[104, 104, 104, 104])?;

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let _ = a.abs();
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(8 * 1260 * 1260, (Kind::Float, Device::Cpu)).reshape(&[8, 1260, 1260]);
    let k = Tensor::randn([80, 8, 4, 4], (Kind::Float, Device::Cpu));

    let now = std::time::Instant::now();
    black_box(for _ in 0..100 {
        let res = a.conv2d(&k, None::<Tensor>, 1, 2, 2, 1);
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
