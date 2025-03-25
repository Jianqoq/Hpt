#[test]
fn test_from_raw() {
    use hpt::backend::Cuda;
    use hpt::ops::ShapeManipulate;
    use hpt::slice;
    use hpt::{ops::Random, Tensor};
    let m = 10;
    let n = 10;
    let a = Tensor::<f32, Cuda>::randn(&[m, n]).expect("failed to create tensor");

    let raw = unsafe {
        a.device()
            .alloc(m * n * 4)
            .expect("failed to alloc raw pointer")
    };

    let c = unsafe { Tensor::<f32, Cuda>::from_raw(raw, &[m, n]) }
        .expect("failed to create tensor from raw pointer");

    let _ = a + c.clone();

    let _sliced_c = slice!(c[0, ..]).expect("failed to slice tensor");

    let reshaped = _sliced_c
        .reshape(&[10, 10])
        .expect("failed to reshape tensor");

    drop(reshaped);
    drop(_sliced_c);
    let (_, _) = unsafe { c.forget().expect("failed to forget tensor") };
}
