use hpt::{
    buitin_templates::cpu::gemm,
    ops::{Matmul, ShapeManipulate, TensorCreator},
    Tensor,
};

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0, 512 * 512)?.reshape(&[512, 512])?;
    let b = Tensor::<f32>::arange(0, 512 * 512)?.reshape(&[512, 512])?;
    let now = std::time::Instant::now();
    for _ in 0..10000 {
        let c = gemm(&a, &b, None)?;
        // println!("{}", c);
    }
    println!("{:?}", now.elapsed() / 10000);
    let now = std::time::Instant::now();
    for _ in 0..10000 {
        let real_res = a.matmul(&b)?;
        // println!("{}", real_res);
    }
    println!("{:?}", now.elapsed() / 10000);
    Ok(())
}
