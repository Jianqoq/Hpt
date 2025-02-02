use rayon::iter::ParallelIterator;
use tensor_dyn::{
    ParStridedIteratorSimdZip, ParStridedIteratorZip, Random, Tensor, TensorCreator, TensorError,
    TensorIterator,
};

fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([2, 4, 6, 8])?;
    let mut b = Tensor::<f32>::empty([2, 4, 6, 8])?;

    b.par_iter_mut().zip(a.par_iter()).for_each(|(b, a)| {
        *b = a;
    });
    println!("{}", b);

    let res = b
        .par_iter()
        .zip(a.par_iter())
        .strided_map(|(res, (b, a))| *res = b + a)
        .collect::<Tensor<f32>>();
    println!("{}", res);

    let res = b
        .par_iter_simd()
        .zip(a.par_iter_simd())
        .strided_map_simd(
            |(res, (b, a))| *res = b + a,
            |(res, (b, a))| res.write_unaligned(b + a),
        )
        .collect::<Tensor<f32>>();
    println!("{}", res);

    Ok(())
}
