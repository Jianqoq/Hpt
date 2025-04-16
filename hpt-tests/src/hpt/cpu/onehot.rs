#[test]
fn onehot() -> anyhow::Result<()> {
    use crate::{TestTypes, TCH_TEST_TYPES};
    use hpt::common::cpu::TensorLike;
    use hpt::ops::AdvancedOps;
    use hpt::ops::TensorCreator;
    use hpt::types::TypeCommon;
    use rand::Rng;
    let mut rng = rand::rng();
    for _ in 0..100 {
        let ndim = rng.random_range(1..=3);
        let shape = (0..ndim)
            .map(|_| rng.random_range(1..=5))
            .collect::<Vec<_>>();
        let depth = rng.random_range(1..=5);
        let tch_input =
            tch::Tensor::randint_low(0, depth, &shape, (TCH_TEST_TYPES, tch::Device::Cpu));
        let mut input = hpt::Tensor::<TestTypes>::empty(&shape)?;
        input.as_raw_mut().copy_from_slice(unsafe {
            std::slice::from_raw_parts(tch_input.data_ptr() as *const TestTypes, tch_input.numel())
        });
        let output = input
            .onehot(depth as usize, ndim - 1, TestTypes::ONE, TestTypes::ZERO)?
            .astype::<TestTypes>()?;
        let tch_output = tch_input.onehot(depth);
        let tch_output_raw = unsafe {
            core::slice::from_raw_parts(
                tch_output.data_ptr() as *const TestTypes,
                tch_output.numel(),
            )
        };
        let hpt_raw = output.as_raw();
        assert_eq!(hpt_raw, tch_output_raw);
    }
    Ok(())
}
