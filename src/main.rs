use tensor_dyn::*;

fn main() -> anyhow::Result<()> {
    set_global_display_lr_elements(1000);
    let a = Tensor::<f32, Cuda, 0>::arange(0, 10 * 10)?.reshape([10, 10])?;
    let b = a.shrinkage(0.0, 1.0)?;
    println!("{}", b);
    Ok(())
}
