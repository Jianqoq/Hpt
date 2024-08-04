use tensor_common::slice;
use tensor_dyn::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_traits::*;
use tensor_dyn::slice::SliceOps;
use tensor_macros::match_selection;
use tensor_common::slice::Slice;
use rayon::iter::ParallelIterator;

fn main() -> anyhow::Result<()> {
    let a = Tensor::<f32>::arange(0.0, 9.0)?.reshape(&[3, 3])?;

    let b = Tensor::<f32>::ones(&[5, 5])?;
    
    let sliced_b = slice!(b[1:4, 1:4])?;
    println!("{:?}", sliced_b);
    sliced_b.iter_mut().zip(a.iter()).for_each(|(x, y)| {
        *x = y;
    });

    println!("{:?}", b);
    Ok(())
}
