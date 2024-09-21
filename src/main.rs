
use tensor_dyn::ShapeManipulate;
use tensor_dyn::{tensor_base::_Tensor, TensorCreator};
use tensor_dyn::NormalReduce;

fn main() {
    let a = _Tensor::<f32>::arange(0, 1024 * 1024 * 8)
        .unwrap()
        .reshape(&[1024, 1024, 8])
        .unwrap();
    let b = _Tensor::<i32>::arange(0, 1024 * 1024 * 8)
    .unwrap()
    .reshape(&[1024, 1024, 8])
    .unwrap();

    let c = a + b;
}
