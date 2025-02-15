use hpt::*;

fn main() -> anyhow::Result<()> {
    let x = Tensor::<f64>::new(&[1f64, 2., 3.]);

    let res = x.par_iter_simd().strided_map_simd(
        |(res, x)| {
            *res = x.sin();
        },
        |(res, x)| {
            res.write_unaligned(x._sin());
        },
    ).collect::<Tensor<f64>>();

    println!("{}", res);
    Ok(())
}
