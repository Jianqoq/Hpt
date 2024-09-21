use tensor_dyn::{StridedIteratorMap, StridedIteratorZip};

fn main() -> anyhow::Result<()> {
    use tensor_dyn::tensor::Tensor;
    use tensor_dyn::TensorIterator;
    let a = Tensor::<f64>::new([0.0, 1.0, 2.0, 3.0]);
    let res = a
        .iter()
        .zip(a.iter())
        .map(|(a, b)| a + b)
        .collect::<Tensor<f64>>();
    println!("{:?}", res);
    Ok(())
}
