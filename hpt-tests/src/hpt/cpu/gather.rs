#[test]
fn test_gather() -> anyhow::Result<()> {
    // use hpt::AdvancedOps;
    // use hpt::TensorCreator;
    // use hpt::TensorLike;
    // use rand::Rng;
    // let mut rng = rand::rng();
    // let ndim = rng.random_range(1..=3);
    // for _ in 0..100 {
    //     let dim_size = rng.random_range(2..=8);
    //     let shape = (0..ndim).map(|_| dim_size).collect::<Vec<_>>();
    //     let input = tch::Tensor::randn(&shape, (tch::Kind::Float, tch::Device::Cpu));
    //     let mut hpt_input = hpt::Tensor::<f32>::empty(&shape)?;
    //     hpt_input.as_raw_mut().copy_from_slice(unsafe {
    //         std::slice::from_raw_parts(input.data_ptr() as *const f32, input.numel())
    //     });
    //     let indices = tch::Tensor::randint_low(
    //         0,
    //         dim_size - 1,
    //         &shape,
    //         (tch::Kind::Int64, tch::Device::Cpu),
    //     );
    //     let mut hpt_indices = hpt::Tensor::<i64>::empty(&shape)?;
    //     hpt_indices.as_raw_mut().copy_from_slice(unsafe {
    //         std::slice::from_raw_parts(indices.data_ptr() as *const i64, indices.numel())
    //     });
    //     let output = hpt_input.gather(&hpt_indices, 0)?;
    //     let tch_output = input.gather(0, &indices, false);
    //     assert_eq!(output.as_raw(), unsafe {
    //         std::slice::from_raw_parts(tch_output.data_ptr() as *const f32, tch_output.numel())
    //     });
    // }
    Ok(())
}
