use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<half::f16, Cuda, 0>::linspace(0, 100, 100, true)?;
    let b = Tensor::<i8, Cuda, 0>::linspace(0, 100, 100, true)?;
    // println!("{}", a);
    let now = std::time::Instant::now();
    let res = a.selu();
    println!("{:?}", now.elapsed());
    println!("{}", res);


    Ok(())
}
