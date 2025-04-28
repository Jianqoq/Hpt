#[test]
fn test_save_load_single() -> anyhow::Result<()> {
    use half::{bf16, f16};
    use hpt::ops::ShapeManipulate;
    use hpt::save_load::{CompressionAlgo, TensorLoader, TensorSaver};
    use hpt::Tensor;
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    let b = Tensor::<i8>::new(&[1, 2, 3, 4, 5]).reshape(&[1, 5])?;
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype::<f16>()?;
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype::<bf16>()?;

    let tmp_dir = std::env::temp_dir();
    let tmp_file = tmp_dir.join("single_test_saver");
    let saver = TensorSaver::new(&tmp_file);
    saver
        .push("a", a.clone(), CompressionAlgo::Zlib, 1)
        .push("b", b.clone(), CompressionAlgo::Deflate, 1)
        .push("c", c.clone(), CompressionAlgo::NoCompression, 1)
        .push("d", d.clone(), CompressionAlgo::Gzip, 1)
        .save()?;

    let mut loaded = TensorLoader::new(&tmp_file)
        .push::<f32>("a", &[])
        .push::<i8>("b", &[])
        .push::<f16>("c", &[])
        .push::<bf16>("d", &[])
        .load()?;

    assert_eq!(a, loaded.remove("a").unwrap().into());
    assert_eq!(b, loaded.remove("b").unwrap().into());
    assert_eq!(c, loaded.remove("c").unwrap().into());
    assert_eq!(d, loaded.remove("d").unwrap().into());

    std::fs::remove_file(&tmp_file)?;
    Ok(())
}

#[test]
fn test_save_load_struct() -> anyhow::Result<()> {
    use half::{bf16, f16};
    use hpt::ops::ShapeManipulate;
    use hpt::Tensor;
    use hpt::{Load, Save};

    #[derive(Save, Load)]
    struct TestStruct {
        a: Tensor<f32>,
        b: Tensor<i8>,
        c: Tensor<f16>,
        d: Tensor<bf16>,
    }
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    let b = Tensor::<i8>::new(&[1, 2, 3, 4, 5]).reshape(&[1, 5])?;
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype::<f16>()?;
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype::<bf16>()?;

    let test_struct = TestStruct { a, b, c, d };

    let tmp_dir = std::env::temp_dir();
    let tmp_file = tmp_dir.join("struct_test_saver");

    test_struct.save(&tmp_file)?;

    let test_struct_loaded = TestStruct::load(&tmp_file)?;

    assert_eq!(test_struct.a, test_struct_loaded.a);
    assert_eq!(test_struct.b, test_struct_loaded.b);
    assert_eq!(test_struct.c, test_struct_loaded.c);
    assert_eq!(test_struct.d, test_struct_loaded.d);

    std::fs::remove_file(&tmp_file)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_save_load_single_cuda() -> anyhow::Result<()> {
    use half::{bf16, f16};
    use hpt::backend::Cuda;
    use hpt::ops::ShapeManipulate;
    use hpt::save_load::{CompressionAlgo, TensorLoader, TensorSaver};
    use hpt::Tensor;
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])?
        .to_cuda::<0>()?;
    let b = Tensor::<i8>::new(&[1, 2, 3, 4, 5])
        .reshape(&[1, 5])?
        .to_cuda::<0>()?;
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .astype::<f16>()?
        .to_cuda::<0>()?;
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .astype::<bf16>()?
        .to_cuda::<0>()?;

    let tmp_dir = std::env::temp_dir();
    let tmp_file = tmp_dir.join("test_saver");
    let saver = TensorSaver::new(&tmp_file);
    saver
        .push("a", a.clone(), CompressionAlgo::Zlib, 1)
        .push("b", b.clone(), CompressionAlgo::Deflate, 1)
        .push("c", c.clone(), CompressionAlgo::NoCompression, 1)
        .push("d", d.clone(), CompressionAlgo::Gzip, 1)
        .save()?;

    let a_loaded = TensorLoader::new(&tmp_file)
        .push("a", &[])
        .load::<Tensor<f32, Cuda>>()?;

    let b_loaded = TensorLoader::new(&tmp_file)
        .push("b", &[])
        .load::<Tensor<i8, Cuda>>()?;

    let c_loaded = TensorLoader::new(&tmp_file)
        .push("c", &[])
        .load::<Tensor<f16, Cuda>>()?;

    let d_loaded = TensorLoader::new(&tmp_file)
        .push("d", &[])
        .load::<Tensor<bf16, Cuda>>()?;

    assert_eq!(a, a_loaded["a"]);
    assert_eq!(b, b_loaded["b"]);
    assert_eq!(c, c_loaded["c"]);
    assert_eq!(d, d_loaded["d"]);

    std::fs::remove_file(&tmp_file)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_save_load_struct_cuda() -> anyhow::Result<()> {
    use half::{bf16, f16};
    use hpt::backend::Cuda;
    use hpt::ops::ShapeManipulate;
    use hpt::Tensor;
    use hpt::{Load, Save};

    #[derive(Save, Load)]
    struct TestStruct {
        a: Tensor<f32, Cuda>,
        b: Tensor<i8, Cuda>,
        c: Tensor<f16, Cuda>,
        d: Tensor<bf16, Cuda>,
    }
    let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0])
        .reshape(&[2, 2])?
        .to_cuda::<0>()?;
    let b = Tensor::<i8>::new(&[1, 2, 3, 4, 5])
        .reshape(&[1, 5])?
        .to_cuda::<0>()?;
    let c = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .astype::<f16>()?
        .to_cuda::<0>()?;
    let d = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .astype::<bf16>()?
        .to_cuda::<0>()?;

    let test_struct = TestStruct { a, b, c, d };

    let tmp_dir = std::env::temp_dir();
    let tmp_file = tmp_dir.join("test_saver");

    test_struct.save(&tmp_file)?;

    let test_struct_loaded = TestStruct::load(&tmp_file)?;

    assert_eq!(test_struct.a, test_struct_loaded.a);
    assert_eq!(test_struct.b, test_struct_loaded.b);
    assert_eq!(test_struct.c, test_struct_loaded.c);
    assert_eq!(test_struct.d, test_struct_loaded.d);

    std::fs::remove_file(&tmp_file)?;
    Ok(())
}
