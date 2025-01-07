use std::fs::File;

use safetensors::SafeTensors;
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};
use tensor_common::shape::shape::Shape;
use tensor_dyn::*;
use traits::SimdMath;

type F32Simd = <f32 as TypeCommon>::Vec;

#[derive(Save, Load)]
struct Conv2d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
}

struct BatchNorm {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    running_mean: Tensor<f32>,
    running_var: Tensor<f32>,
    eps: f32,
}

impl Conv2dBatchNorm {
    #[track_caller]
    pub fn forward(
        &self,
        x: &Tensor<f32>,
        activation: fn(F32Simd) -> F32Simd,
    ) -> anyhow::Result<Tensor<f32>> {
        Ok(x.batchnorm_conv2d(
            &self.weight,
            &self.running_mean,
            &self.running_var,
            &self.running_gamma,
            &self.running_beta,
            self.bias.as_ref(),
            self.eps,
            [self.steps as i64, self.steps as i64],
            [
                (self.padding as i64, self.padding as i64),
                (self.padding as i64, self.padding as i64),
            ],
            [self.dilation as i64, self.dilation as i64],
            Some(activation),
        )?)
    }
}

#[derive(Save, Load)]
pub struct MaxPool2d {
    kernel_size: Shape,

    stride: usize,

    padding: usize,

    dilation: usize,
}

impl MaxPool2d {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(x.maxpool2d(
            &self.kernel_size,
            [self.stride as i64, self.stride as i64],
            [
                (self.padding as i64, self.padding as i64),
                (self.padding as i64, self.padding as i64),
            ],
            [self.dilation as i64, self.dilation as i64],
        )?)
    }
}

#[derive(Save, Load)]
pub struct AdaptiveAvgPool2d {
    kernel_size: [i64; 2],
}

impl AdaptiveAvgPool2d {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(x.adaptive_avgpool2d(self.kernel_size)?)
    }
}

#[derive(Save, Load)]
pub struct DownSample {
    conv: Conv2dBatchNorm,
}

impl DownSample {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(self.conv.forward(x, |x| x)?)
    }
}

#[derive(Save, Load)]
pub struct Sequential {
    layers: Vec<BasicBlock>,
}

impl Sequential {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Save, Load)]
pub struct BasicBlock {
    bn_conv1: Conv2dBatchNorm,
    bn_conv2: Conv2dBatchNorm,
    downsample: Option<DownSample>,
}

impl BasicBlock {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let identity = if let Some(downsample) = &self.downsample {
            downsample.forward(&x)?
        } else {
            x.clone()
        };
        let out = self.bn_conv1.forward(&x, |x| x.relu())?;
        let out = self.bn_conv2.forward(&out, |x| x)?;
        out.par_iter_mut_simd()
            .zip(identity.par_iter_simd())
            .for_each(
                |(a, b)| *a = (*a + b)._relu(),
                |(a, b)| {
                    a.write_unaligned((a.read_unaligned() + b).relu());
                },
            );
        Ok(out)
    }
}

impl Linear {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out.add_(&self.bias, &out)?)
    }
}

#[derive(Save, Load)]
pub struct Conv2dBatchNorm {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    running_mean: Tensor<f32>,
    running_var: Tensor<f32>,
    running_gamma: Tensor<f32>,
    running_beta: Tensor<f32>,
    eps: f32,
    steps: usize,
    padding: usize,
    dilation: usize,
}

#[derive(Save, Load)]
pub struct Linear {
    weight: Tensor<f32>,

    bias: Tensor<f32>,
}

#[derive(Save, Load)]
pub struct ResNet {
    conv1: Conv2dBatchNorm,
    max_pool1: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avg_pool: AdaptiveAvgPool2d,
    fc: Linear,
}

impl ResNet {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let x = self.conv1.forward(&x, |x| x.relu())?;
        let x = self.max_pool1.forward(&x)?;
        let x = self.layer1.forward(&x)?;
        let x = self.layer2.forward(&x)?;
        let x = self.layer3.forward(&x)?;
        let x = self.layer4.forward(&x)?;
        let x = self.avg_pool.forward(&x)?;
        let x = self.fc.forward(&x)?;
        Ok(x)
    }
}

fn convert_keys_to_json(safetensor: &SafeTensors) -> anyhow::Result<Value> {
    let mut root = serde_json::Map::new();

    for (key, tensor) in safetensor.tensors() {
        let shape = tensor.shape();
        let dtype = tensor.dtype();

        insert_nested_key(&mut root, &key, &dtype, shape)?;
    }

    Ok(Value::Object(root))
}

// 插入嵌套键到 JSON 结构中
fn insert_nested_key(
    root: &mut serde_json::Map<String, Value>,
    key: &str,
    dtype: &safetensors::Dtype,
    shape: &[usize],
) -> anyhow::Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = root;

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            // 最后一个部分，插入数据
            current.insert(
                part.to_string(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "name": key,
                }),
            );
        } else {
            // 中间部分，构建嵌套对象
            current = current
                .entry(part.to_string())
                .or_insert_with(|| json!({}))
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("Expected object at {}", part))?;
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let buffer = std::fs::read("model.safetensors")?;
    let safetensor = SafeTensors::deserialize(&buffer)?;
    let json = convert_keys_to_json(&safetensor)?;
    println!("{:#?}", json);
    // let resnet = create_resnet();
    // resnet.save("resnet.model")?;
    // let data = ResNet::load("resnet.model")?;
    // let input = Tensor::<f32>::randn(&[5, 128, 128, 3])?;
    // let now = std::time::Instant::now();
    // // for _ in 0..10 {
    // let output = data.forward(&input)?;
    // // }
    // println!("time: {:?}", now.elapsed() / 10);
    // sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    // if let Some(process) = sys.process(pid) {
    //     println!("After Inference - Memory usage: {} KB", process.memory());
    // }
    Ok(())
}
