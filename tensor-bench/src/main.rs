extern crate tch;
use tch::Tensor;

fn main() {
    // 创建一个随机张量
    let tensor = Tensor::randn(&[3, 3], (tch::Kind::Float, tch::Device::Cpu));
    println!("随机张量: {:?}", tensor);

    // 执行矩阵乘法
    let result = tensor.matmul(&tensor);
    println!("矩阵乘法结果: {:?}", result);
}