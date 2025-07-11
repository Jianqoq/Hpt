use hpt::{
    error::TensorError,
    ops::Random,
    save_load::{CompressionAlgo, TensorLoader, TensorSaver},
    Tensor,
};
fn main() -> Result<(), TensorError> {
    let a = Tensor::<f32>::randn([10, 10])?;
    let b = Tensor::<f64>::randn([10, 10])?;
    let c = &a + &b;

    // path need to convert to std::path::PathBuf
    let saver = TensorSaver::new("f.ftz");

    saver
        .push("a", a, CompressionAlgo::Gzip, 9)
        .push("b", b, CompressionAlgo::Gzip, 9)
        .push("c", c, CompressionAlgo::Gzip, 9)
        .save()
        .expect("save failed");

    let loader = TensorLoader::new("f.ftz");

    /* load all must make sure all the tensors are the same type, because we saved different types in this example, so we can't load all at once
    let datas = loader
        .load_all::<f32, Tensor<f32>, { std::mem::size_of::<f32>() }>()
        .expect("load failed");
    */

    let res = loader
        .push::<f32>("b", &[])
        .push::<f32>("c", &[])
        .load()
        .expect("load failed");

    Ok(())
}
