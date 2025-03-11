use hpt::{buitin_templates::cpu::gemm, ops::{TensorCreator, ShapeManipulate, Matmul}, Tensor};



fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0, 512 * 512)?.reshape(&[512, 512])?;
    let b = Tensor::<f32>::arange(0, 512 * 512)?.reshape(&[512, 512])?;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let c = gemm(&a, &b, None)?;
    }
    println!("{:?}", now.elapsed() / 100);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let real_res = a.matmul(&b)?;
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
