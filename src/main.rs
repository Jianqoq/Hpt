use hpt::{
    ops::{Gemm, Matmul, TensorCreator},
    types::f16,
    Tensor,
};
fn main() -> anyhow::Result<()> {
    let n = 614;
    let m = 611;
    let k = 1490;
    let a = Tensor::<f16>::ones(&[n, k])?;
    let b = Tensor::<f16>::ones(&[k, m])?;
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = a.matmul(&b)?;
        println!("{}", res);
    }
    println!("{:?}", now.elapsed() / 100);
    let now = std::time::Instant::now();
    for _ in 0..1 {
        let res = a.gemm(
            &b,
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            false,
            false,
            false,
        )?;
        println!("{}", res);
    }
    println!("{:?}", now.elapsed() / 100);

    Ok(())
}
