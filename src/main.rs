use hpt::*;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]).to_cuda::<0>()?;
    let y = Tensor::<i64>::new(&[4i64, 5, 6]).to_cuda::<0>()?;

    let result = x + y; // with `normal_promote` feature enabled, i64 + f64 will output f64
    println!("{}", result); // [5., 7., 9.]

    Ok(())
}
