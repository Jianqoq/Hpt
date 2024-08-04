use tensor_dyn::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_traits::*;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0.0, 270.0)?.reshape(&[45, 2, 3])?;

    let mut grid = _Tensor::<f32>::affine_grid(&a, &[45, 5, 3, 4], false)?;
    // let now = std::time::Instant::now();
    // for _ in 0..1000 {
    //     grid = _Tensor::<f32>::affine_grid(&a, &[50, 5, 3, 4], false)?;
    // }
    // println!("{:?}", now.elapsed());
    println!("{:?}", grid);
    Ok(())
}
