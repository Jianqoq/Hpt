use std::hint::black_box;
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
    black_box(for _ in 0..1 {
        let _ = a.relu6();
    });
    println!("hpt time: {:?}", now.elapsed() / 100);

    let a = Tensor::arange(8 * 2048 * 2048, (Kind::Float, Device::Cpu)).reshape(&[8, 2048, 2048]);
    // let c = Tensor::randn([104, 1, 104, 1], (Kind::Float, Device::Cpu));

    let now = std::time::Instant::now();
    black_box(for _ in 0..1 {
        let (res1, res2) = a.topk(10, 1, true, false);
        println!("{}", res1);
    });
    println!("torch time: {:?}", now.elapsed() / 100);
    Ok(())
}
