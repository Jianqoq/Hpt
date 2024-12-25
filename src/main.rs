use std::io::BufRead;

use tensor_common::slice;
use tensor_dyn::*;


struct Conv2dBatchNorm {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    gamma: Tensor<f32>,
    beta: Tensor<f32>,
    mean: Tensor<f32>,
    var: Tensor<f32>,
    epsilon: f32,
}

impl Conv2dBatchNorm {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct Relu;

impl Relu {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct MaxPool2d;

impl MaxPool2d {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct BasicBlock {
    conv2d_bn: Conv2dBatchNorm,
    relu: Relu,
    conv2d_bn2: Conv2dBatchNorm,
    downsample: Conv2dBatchNorm,
}

impl BasicBlock {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct AvgPool2d;

impl AvgPool2d {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct Sequential {
    layers: Vec<BasicBlock>,
}

impl Sequential {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

struct ResNet {
    conv2d_bn: Conv2dBatchNorm,
    relu: Relu,
    maxpool2d: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avgpool2d: AvgPool2d,
}

impl ResNet {
    fn new() -> Self {
        todo!()
    }
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        todo!()
    }
}

fn main() -> anyhow::Result<()> {
    let loader: Vec<Tensor<f32>> = TensorLoader::new("test.ftz".into())
        .push("test", &[])
        .load()?;

    println!("{}", loader[0]);

    Ok(())
}

// fn visit_dirs(dir: &str) -> std::io::Result<()> {
//     let mut saver = TensorSaver::new("weights/data.ftz".into());
//     // 读取当前目录的条目
//     for entry_result in std::fs::read_dir(dir)? {
//         let entry = entry_result?;
//         let file_name = entry.file_name();
//         let path = entry.path();

//         // 如果是目录，递归调用
//         if path.is_dir() {
//             visit_dirs(path.to_str().unwrap())?;
//         } else {
//             let file = std::fs::File::open(&path)?;
//             let reader = std::io::BufReader::new(file);
//             let mut lines = reader.lines();
//             let shape = lines.next().unwrap().unwrap();
//             let after_shape = shape.strip_prefix("shape:").unwrap().trim();
//             let mut shape = Vec::new();
//             if !after_shape.is_empty() {
//                 let tokens = after_shape.split_whitespace();
//                 for token in tokens {
//                     // 尝试解析为整数
//                     if let Ok(val) = token.parse::<i64>() {
//                         shape.push(val);
//                     }
//                 }
//             }
//             println!("{}: {:?}", file_name.to_str().unwrap(), shape);
//             let mut data = Vec::new();
//             while let Some(line) = lines.next() {
//                 let line = line?;
//                 let tokens = line.split_whitespace();
//                 for token in tokens {
//                     if let Ok(val) = token.parse::<f32>() {
//                         data.push(val);
//                     }
//                 }
//             }
//             if shape.len() > 0 {
//                 let tensor = Tensor::<f32>::new(data.clone()).reshape(shape).unwrap();
//                 saver = saver.push(
//                     file_name.to_str().unwrap(),
//                     tensor,
//                     CompressionAlgo::Gzip,
//                     Endian::Native,
//                     9,
//                 );
//             }
//         }
//     }
//     saver.save()?;
//     Ok(())
// }

// fn load_dirs(dir: &str) -> std::io::Result<()> {
//     // 读取当前目录的条目
//     for entry_result in std::fs::read_dir(dir)? {
//         let entry = entry_result?;
//         let file_name = entry.file_name();
//         let path = entry.path();

//         // 如果是目录，递归调用
//         if path.is_dir() {
//             visit_dirs(path.to_str().unwrap())?;
//         } else {
//             let file = std::fs::File::open(&path)?;
//             let reader = std::io::BufReader::new(file);
//             let mut lines = reader.lines();
//             let shape = lines.next().unwrap().unwrap();
//             let after_shape = shape.strip_prefix("shape:").unwrap().trim();
//             let mut shape = Vec::new();
//             if !after_shape.is_empty() {
//                 let tokens = after_shape.split_whitespace();
//                 for token in tokens {
//                     // 尝试解析为整数
//                     if let Ok(val) = token.parse::<i64>() {
//                         shape.push(val);
//                     }
//                 }
//             }
//             println!("{}: {:?}", file_name.to_str().unwrap(), shape);
//             let mut data = Vec::new();
//             while let Some(line) = lines.next() {
//                 let line = line?;
//                 let tokens = line.split_whitespace();
//                 for token in tokens {
//                     if let Ok(val) = token.parse::<f32>() {
//                         data.push(val);
//                     }
//                 }
//             }
//             if shape.len() > 0 {
//                 let loader = TensorLoader::new(r"C:\Users\123\eTensor-1\weights\data.ftz".into());
//                 let res = loader
//                     .push(file_name.to_str().unwrap(), &[])
//                     .load::<f32, Tensor<f32>, 4>()?;
//                 println!("{}", res[0]);
//                 std::thread::sleep(std::time::Duration::from_millis(1000));
//             }
//         }
//     }
//     Ok(())
// }
