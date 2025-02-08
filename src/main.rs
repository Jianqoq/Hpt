use hpt_core::{FloatOutPooling, Random, Shape, Tensor, TensorError};
fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([1, 100, 100, 100])?;
    let b = a.avgpool2d(Shape::new([3, 3]), [2, 2], [(0, 0), (0, 0)], [1, 1])?;
    println!("{:?}", b);
    Ok(())
}
