use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let m = 10000;
    let n = 10000;
    let k = 10000;
    let a = Tensor::<f32, Cuda, 0>::arange(0, m * k)?.reshape([m, k])?;
    let b = Tensor::<f32, Cuda, 0>::arange(0, k * n)?.reshape([k, n])?;
    let now = std::time::Instant::now();
    let c1 = a.matmul_blocked(&b)?;
    c1.device().synchronize()?;
    println!("{:?}", now.elapsed());
    println!("{}", c1);

    let now = std::time::Instant::now();
    let c2 = a.matmul(b)?;
    c2.device().synchronize()?;
    println!("{:?}", now.elapsed());
    println!("{}", c2);

    // let a = Tensor::<f32, Cpu, 0>::arange(0, m * k)?.reshape([m, k])?;
    // let b = Tensor::<f32, Cpu, 0>::arange(0, k * n)?.reshape([k, n])?;

    // let now = std::time::Instant::now();
    // let c = a.matmul(&b)?;
    // println!("{:?}", now.elapsed());
    // println!("{}", c);

    let c1_cpu = c1.to_cpu()?;
    let c2_cpu = c2.to_cpu()?;
    println!("{}", c1_cpu.allclose(&c2_cpu));
    Ok(())
}
