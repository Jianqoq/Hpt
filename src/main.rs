use hpt::{
    buitin_templates::cpu::gemm,
    ops::{Matmul, ShapeManipulate, TensorCreator},
    utils::set_display_elements,
    Tensor,
};

fn main() -> anyhow::Result<()> {
    let m = 7;
    let n = 2;
    let k = 2;
    let a = Tensor::<f32>::arange(0, m * k)?.reshape(&[m, k])?;
    println!("a: {}", a);
    let b = Tensor::<f32>::arange(0, k * n)?.reshape(&[k, n])?;
    let c = gemm(&a, &b, None)?;
    let real_res = a.matmul(&b)?;

    println!("{}", c);
    println!("{}", real_res);
    // assert!(c.allclose(&real_res));
    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let c = gemm(&a, &b, None)?;
    //     // println!("{}", c);
    // }
    // println!("{:?}", now.elapsed() / 10000);
    // let now = std::time::Instant::now();
    // for _ in 0..10000 {
    //     let real_res = a.matmul(&b)?;
    //     // println!("{}", real_res);
    // }
    // println!("{:?}", now.elapsed() / 10000);
    Ok(())
}
