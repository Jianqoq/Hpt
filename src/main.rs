use hpt::WindowOps;

fn main() -> anyhow::Result<()> {
    let a = hpt::Tensor::<f32>::hamming_window(10, true)?;
    println!("{:?}", a);
    Ok(())
}
