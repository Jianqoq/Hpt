#[test]
fn test_from_raw() {
    use hpt::ops::ShapeManipulate;
    use hpt::slice;
    use hpt::{ops::Random, Tensor};
    let m = 10;
    let n = 10;
    let a = Tensor::<f32>::randn(&[m, n]).expect("failed to create tensor");
    let layout = std::alloc::Layout::from_size_align(m * n * 4, 64).unwrap();

    let raw = unsafe { std::alloc::alloc(layout) };
    let c = unsafe { Tensor::<f32>::from_raw(raw as *mut f32, &[m, n]) }
        .expect("failed to create tensor from raw pointer");

    let _ = a + c.clone();

    let _sliced_c = slice!(c[0, ..]).expect("failed to slice tensor");

    let reshaped = _sliced_c
        .reshape(&[10, 10])
        .expect("failed to reshape tensor");

    drop(reshaped);
    drop(_sliced_c);
    let (raw, _) = unsafe { c.forget().expect("failed to forget tensor") };

    unsafe { std::alloc::dealloc(raw, layout) };
}
