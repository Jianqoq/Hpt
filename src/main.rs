use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(1000);
    let a = Tensor::<f32, Cuda, 0>::arange(0, 10 * 10)?.reshape([10, 10])?;
    let b = a.pad(&[(5, 5), (5, 5)], 0.0)?;
    println!("{}", b);

    let a = Tensor::<f32, Cpu, 0>::arange(0, 10 * 10)?.reshape([10, 10])?;
    let b = a.pad(&[(5, 5), (5, 5)], 0.0)?;
    println!("{}", b);
    Ok(())
}
