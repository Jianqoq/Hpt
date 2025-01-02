use std::fs::File;

use tensor_common::shape::Shape;
use tensor_dyn::*;
use traits::SimdMath;

type F32Simd = <f32 as TypeCommon>::Vec;

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

impl Conv2dBatchNorm {
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
pub struct AvgPool2d {
    kernel_size: Shape,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl AvgPool2d {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(x.avgpool2d(
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
        let x = if let Some(downsample) = &self.downsample {
            downsample.forward(&x)?
        } else {
            x.clone()
        };
        let out = self.bn_conv1.forward(&x, |x| x.relu())?;
        let out = self.bn_conv2.forward(&out, |x| x)?;
        Ok((x + out).relu()?)
    }
}

#[derive(Save, Load)]
pub struct Linear {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out + &self.bias)
    }
}

#[derive(Save, Load)]
pub struct ResNet {
    bn_conv1: Conv2dBatchNorm,
    max_pool1: MaxPool2d,
    layer1: Sequential,
    layer2: Sequential,
    layer3: Sequential,
    layer4: Sequential,
    avg_pool: AvgPool2d,
    fc: Linear,
}

impl ResNet {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let x = self.bn_conv1.forward(&x, |x| x)?;
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

fn main() -> anyhow::Result<()> {
    let resnet = ResNet::load("resnet.bin")?;
    Ok(())
}
