
use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<i16, Cpu, 0>::zeros([10, 10])?;
    let b = Tensor::<i16, Cpu, 0>::zeros([10, 10])?;
    let now = std::time::Instant::now();
    let res = a.nansum([0, 1], false)?;
    println!("{:?}", now.elapsed());
    println!("{}", res);
    Ok(())
}