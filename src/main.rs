use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<half::f16, Cuda, 0>::linspace(0, 100, 100, true)?;
    // println!("{}", a);
    let now = std::time::Instant::now();
    let res = a.erf()?;
    println!("{:?}", now.elapsed());
    println!("{}", res);
    
    Ok(())
}
