use tensor_dyn::{ set_global_display_precision, TensorCreator };
use tensor_dyn::tensor_base::_Tensor;
use tensor_dyn::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    set_global_display_precision(7);
    let a = _Tensor::<f32>::randn(&[8 * 2048 * 512, 2, 3])?;
    let mut i = 0;
    let now = std::time::Instant::now();
    for _ in 0..100 {
        let _ = a.affine_grid(&[8 * 2048 * 512, 2, 8, 3], true)?;
        i += 1;
    }
    println!("{:?}", now.elapsed() / 100);
    println!("{}", i);
    Ok(())
}
