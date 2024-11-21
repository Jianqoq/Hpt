use tch::Tensor;
use tensor_dyn::{ tensor_base::_Tensor, TensorCreator };
use tensor_dyn::ShapeManipulate;
use tensor_dyn::TensorLike;
use tensor_dyn::TensorInfo;

#[allow(unused)]
fn assert_eq(b: &_Tensor<i64>, a: &Tensor) {
    let a_raw = unsafe { std::slice::from_raw_parts(a.data_ptr() as *const i64, b.size()) };
    let b_raw = b.as_raw();

    for i in 0..b.size() {
        if a_raw[i] != b_raw[i] {
            panic!("{} != {}, bytes: {:?}, {:?}", a_raw[i], b_raw[i], a_raw[i].to_ne_bytes(), b_raw[i].to_ne_bytes());
        }
    }
}

#[test]
fn test_cumsum_1dim() -> anyhow::Result<()> {
    let a = _Tensor::<f64>::arange(0, 10)?;
    let b = a.cumsum(0)?;
    let tch_a = tch::Tensor::arange(10, (tch::Kind::Double, tch::Device::Cpu));
    let tch_b = tch_a.cumsum(0, None);
    println!("{}", b);
    println!("{}", tch_b);
    Ok(())
}

#[test]
fn test_cumsum_2dim() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 100)?.reshape(&[10, 10])?;
    println!("{}", a);
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&[10, 10]);
    let b = a.cumsum(0)?;
    let tch_b = tch_a.cumsum(0, tch::Kind::Int64);
    println!("{}", b);
    assert_eq(&b, &tch_b);
    
    let c = a.cumsum(1)?;
    let tch_c = tch_a.cumsum(1, tch::Kind::Int64);
    assert_eq(&c, &tch_c);

    Ok(())
}

#[test]
fn test_cumsum_2dim_uncontiguous() -> anyhow::Result<()> {
    let a = _Tensor::<i64>::arange(0, 100)?.reshape(&[10, 10])?;
    let a = a.permute([1, 0])?;
    let tch_a = tch::Tensor::arange(100, (tch::Kind::Int64, tch::Device::Cpu)).reshape(&[10, 10]).permute(&[1, 0][..]);
    let b = a.cumsum(0)?;
    let tch_b = tch_a.cumsum(0, tch::Kind::Int64);
    assert_eq(&b, &tch_b);

    let c = a.cumsum(1)?;
    let tch_c = tch_a.cumsum(1, tch::Kind::Int64);
    assert_eq(&c, &tch_c);

    Ok(())
}