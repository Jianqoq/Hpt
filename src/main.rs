use hpt::{
    buitin_templates::cpu::gemm, ops::{Matmul, TensorCreator}, Tensor
};
fn main() -> anyhow::Result<()> {
    let m = 512 * 1;
    let n = 512 * 1;
    let k = 512 * 1;
    // let a = Tensor::<f32>::arange(0, m * k)?.reshape(&[m, k])?;
    // let b = Tensor::<f32>::arange(0, k * n)?.reshape(&[k, n])?;
    let a = Tensor::<f32>::ones(&[m, k])?;
    let b = Tensor::<f32>::ones(&[k, n])?;
    // println!("a: {}", a);
    // println!("b: {}", b);
    let now = std::time::Instant::now();
    for _ in 0..1000 {
        let c = gemm(&a, &b, None, 16)?;
    }
    println!("{:?}", now.elapsed() / 1000);
    let now = std::time::Instant::now();
    for _ in 0..1000 {
        let real_res = a.matmul(&b)?;
    }
    println!("{:?}", now.elapsed() / 1000);
    Ok(())
}
