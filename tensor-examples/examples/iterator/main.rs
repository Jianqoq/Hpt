use rayon::iter::ParallelIterator;
use tensor_dyn::{
    ParStridedIteratorZip, Random, Tensor, TensorCreator, TensorError, TensorIterator,
};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([2, 4, 6, 8])?;
    let b = Tensor::<f32>::empty([2, 4, 6, 8])?;

    b.par_iter_mut().zip(a.par_iter()).for_each(|(b, a)| {
        *b = a;
    });
    println!("{}", b);

    let res = b
        .par_iter()
        .zip(a.par_iter())
        .strided_map(|(b, a)| b + a)
        .collect::<Tensor<f32>>();
    println!("{}", res);

    Ok(())
}
