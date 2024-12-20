use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32, Cuda, 0>::arange(0, 1000 * 100000)?.reshape([1000, 100000])?;
    let now = std::time::Instant::now();
    let b = a.sum([0, 1], false)?;
    println!("{:?}", now.elapsed());
    println!("{}", b);

    let a = Tensor::<f32, Cpu, 0>::arange(0, 1000 * 100000)?.reshape([1000, 100000])?;
    let now = std::time::Instant::now();
    let b = a.sum([0, 1], false)?;
    println!("{:?}", now.elapsed());
    println!("{}", b);
    Ok(())
}
