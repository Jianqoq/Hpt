fn main() -> anyhow::Result<()> {
    use tensor_dyn::tensor::Tensor;
    use tensor_dyn::ShapeManipulate;
    let tensor1 = Tensor::<f64>::new([1.0, 2.0]).reshape(&[2, 1, 1]).unwrap();
    let tensor2 = Tensor::<f64>::new([3.0, 4.0]).reshape(&[2, 1, 1]).unwrap();
    let stacked_tensor = Tensor::dstack(vec![&tensor1, &tensor2]).unwrap();
    assert!(stacked_tensor.allclose(&Tensor::<f64>::new([[[1.0, 3.0]], [[2.0, 4.0]]])));
    Ok(())
}
