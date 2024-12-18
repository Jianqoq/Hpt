
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<i64, Cuda, 0>::arange(0, 10 * 10 * 10)?.reshape([10, 10, 10])?;
    // println!("{}", a);
    let now = std::time::Instant::now();
    let res = a.argmax([0, 1], false)?;
    println!("{:?}", now.elapsed());
    println!("{}", res);

    let a = Tensor::<i64, Cpu, 0>::arange(0, 10 * 10 * 10)?.reshape([10, 10, 10])?;
    // println!("{}", a);
    let now = std::time::Instant::now();
    let res = a.argmax([0, 1], false)?;
    println!("{:?}", now.elapsed());
    println!("{}", res);
    Ok(())
}