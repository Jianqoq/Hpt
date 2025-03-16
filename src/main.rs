use hpt::{
    buitin_templates::cpu::gemm,
    ops::{Matmul, ShapeManipulate, TensorCreator},
    Tensor,
};

fn main() -> anyhow::Result<()> {
    let m = 512 * 4;
    let n = 512 * 4;
    let k = 512 * 4;
    let a = Tensor::<f32>::arange(0, m * k)?.reshape(&[m, k])?;
    let b = Tensor::<f32>::arange(0, k * n)?.reshape(&[k, n])?;
    // let now = std::time::Instant::now();
    // for _ in 0..100 {
    //     let c = gemm(&a, &b, None)?;
    //     // println!("{}", c);
    // }
    // println!("{:?}", now.elapsed() / 100);
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let real_res = a.matmul(&b)?;
        // println!("{}", real_res);
    }
    println!("{:?}", now.elapsed() / 100);
    Ok(())
}
