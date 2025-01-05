use std::{collections::HashMap, fs::File, path::Path};

use safetensors::SafeTensors;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};
use tensor_common::shape::shape::Shape;
use tensor_dyn::*;
use traits::SimdMath;

type F32Simd = <f32 as TypeCommon>::Vec;

#[derive(Save, Load, FromSafeTensors)]
pub struct Conv2dBatchNorm {
    #[map(from = "prev_layer", current = "weight")]
    weight: Tensor<f32>,
    #[map(from = "prev_layer", current = "bias")]
    bias: Option<Tensor<f32>>,
    #[map(from = "prev_layer", current = "running_mean")]
    running_mean: Tensor<f32>,
    #[map(from = "prev_layer", current = "running_var")]
    running_var: Tensor<f32>,
    #[map(from = "prev_layer", current = "running_gamma")]
    running_gamma: Tensor<f32>,
    #[map(from = "prev_layer", current = "running_beta")]
    running_beta: Tensor<f32>,
    #[map(from = "prev_layer", value = 10.0)]
    eps: f32,
    #[map(from = "prev_layer", value = 10)]
    steps: usize,
    #[map(from = "prev_layer", value = 10)]
    padding: usize,
    #[map(from = "prev_layer", value = 10)]
    dilation: usize,
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

#[derive(Save, Load, FromSafeTensors)]
pub struct MaxPool2d {
    #[map(from = "prev_layer", value = Shape::new([3, 3]))]
    kernel_size: Shape,
    #[map(from = "prev_layer", value = 2)]
    stride: usize,
    #[map(from = "prev_layer", value = 1)]
    padding: usize,
    #[map(from = "prev_layer", value = 1)]
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

#[derive(Save, Load, FromSafeTensors)]
pub struct AdaptiveAvgPool2d {
    #[map(from = "prev_layer", value = [1, 1])]
    kernel_size: [i64; 2],
}

impl AdaptiveAvgPool2d {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(x.adaptive_avgpool2d(self.kernel_size)?)
    }
}

#[derive(Save, Load, FromSafeTensors)]
pub struct DownSample {
    #[map(from = "prev_layer", current = "conv")]
    conv: Conv2dBatchNorm,
}

impl DownSample {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        Ok(self.conv.forward(x, |x| x)?)
    }
}

#[derive(Save, Load, FromSafeTensors)]
pub struct Sequential {
    #[map(from = "prev_layer", vec_len = 4)]
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

#[derive(Save, Load, FromSafeTensors)]
pub struct BasicBlock {
    #[map(from = "prev_layer", current = "bn_conv1")]
    bn_conv1: Conv2dBatchNorm,
    
    #[map(from = "prev_layer", current = "bn_conv2")]
    bn_conv2: Conv2dBatchNorm,
    
    #[map(from = "prev_layer", current = "downsample")]
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

#[derive(Save, Load, FromSafeTensors)]
pub struct Linear {
    #[map(from = "prev_layer", current = "weight")]
    weight: Tensor<f32>,
    #[map(from = "prev_layer", current = "bias")]
    bias: Tensor<f32>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out.add_(&self.bias, &out)?)
    }
}

#[derive(Save, Load, FromSafeTensors)]
pub struct ResNet {
    #[map(from = "prev_layer", current = "conv1")]
    conv1: Conv2dBatchNorm,
    
    #[map(from = "prev_layer", current = "max_pool1")]
    max_pool1: MaxPool2d,
    
    #[map(from = "prev_layer", current = "layer1")]
    layer1: Sequential,
    
    #[map(from = "prev_layer", current = "layer2")]
    layer2: Sequential,
    
    #[map(from = "prev_layer", current = "layer3")]
    layer3: Sequential,
    
    #[map(from = "prev_layer", current = "layer4")]
    layer4: Sequential,
    
    #[map(from = "prev_layer", current = "avg_pool")]
    avg_pool: AdaptiveAvgPool2d,
    
    #[map(from = "prev_layer", current = "fc")]
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

fn create_resnet() -> ResNet {
    let path = Path::new("data.ftz");
    let loader = TensorLoader::new(path.to_path_buf());
    let data = loader
        .load_all::<f32, Tensor<f32>, 4>()
        .expect("failed to load data");
    let conv1_weight = data["conv1_weight.txt"]
        .clone()
        .permute([2, 3, 1, 0])
        .expect("permute failed");
    let bn1_beta = data["bn1_bias.txt"].clone();
    let bn1_running_mean = data["bn1_running_mean.txt"].clone();
    let bn1_running_gamma = data["bn1_weight.txt"].clone();
    let bn1_running_var = data["bn1_running_var.txt"].clone();
    let bn_conv1 = Conv2dBatchNorm {
        weight: conv1_weight,
        bias: None,
        running_mean: bn1_running_mean,
        running_var: bn1_running_var,
        running_gamma: bn1_running_gamma,
        running_beta: bn1_beta,
        eps: 1e-5,
        steps: 2,
        padding: 3,
        dilation: 1,
    };
    let max_pool1 = MaxPool2d {
        kernel_size: Shape::new([3, 3]),
        stride: 2,
        padding: 1,
        dilation: 1,
    };

    fn create_basic_block(
        data: &HashMap<String, Tensor<f32>>,
        weight: &str,
        running_mean: &str,
        running_var: &str,
        running_gamma: &str,
        running_beta: &str,
        conv2_weight: &str,
        conv2_running_mean: &str,
        conv2_running_var: &str,
        conv2_running_gamma: &str,
        conv2_running_beta: &str,
        eps: f32,
        conv1_steps: usize,
        conv1_padding: usize,
        conv1_dilation: usize,
        conv2_steps: usize,
        conv2_padding: usize,
        conv2_dilation: usize,
    ) -> BasicBlock {
        BasicBlock {
            bn_conv1: Conv2dBatchNorm {
                weight: data[weight]
                    .clone()
                    .permute([2, 3, 1, 0])
                    .expect("permute failed"),
                bias: None,
                running_mean: data[running_mean].clone(),
                running_var: data[running_var].clone(),
                running_gamma: data[running_gamma].clone(),
                running_beta: data[running_beta].clone(),
                eps,
                steps: conv1_steps,
                padding: conv1_padding,
                dilation: conv1_dilation,
            },
            bn_conv2: Conv2dBatchNorm {
                weight: data[conv2_weight]
                    .clone()
                    .permute([2, 3, 1, 0])
                    .expect("permute failed"),
                bias: None,
                running_mean: data[conv2_running_mean].clone(),
                running_var: data[conv2_running_var].clone(),
                running_gamma: data[conv2_running_gamma].clone(),
                running_beta: data[conv2_running_beta].clone(),
                eps,
                steps: conv2_steps,
                padding: conv2_padding,
                dilation: conv2_dilation,
            },
            downsample: None,
        }
    }

    fn create_basic_block_with_downsample(
        data: &HashMap<String, Tensor<f32>>,
        weight: &str,
        running_mean: &str,
        running_var: &str,
        running_gamma: &str,
        running_beta: &str,
        conv2_weight: &str,
        conv2_running_mean: &str,
        conv2_running_var: &str,
        conv2_running_gamma: &str,
        conv2_running_beta: &str,
        downsample_weight: &str,
        downsample_running_mean: &str,
        downsample_running_var: &str,
        downsample_running_gamma: &str,
        downsample_running_beta: &str,
        eps: f32,
        conv1_steps: usize,
        conv1_padding: usize,
        conv1_dilation: usize,
        conv2_steps: usize,
        conv2_padding: usize,
        conv2_dilation: usize,
        downsample_steps: usize,
        downsample_padding: usize,
        downsample_dilation: usize,
    ) -> BasicBlock {
        BasicBlock {
            bn_conv1: Conv2dBatchNorm {
                weight: data[weight]
                    .clone()
                    .permute([2, 3, 1, 0])
                    .expect("permute failed"),
                bias: None,
                running_mean: data[running_mean].clone(),
                running_var: data[running_var].clone(),
                running_gamma: data[running_gamma].clone(),
                running_beta: data[running_beta].clone(),
                eps,
                steps: conv1_steps,
                padding: conv1_padding,
                dilation: conv1_dilation,
            },
            bn_conv2: Conv2dBatchNorm {
                weight: data[conv2_weight]
                    .clone()
                    .permute([2, 3, 1, 0])
                    .expect("permute failed"),
                bias: None,
                running_mean: data[conv2_running_mean].clone(),
                running_var: data[conv2_running_var].clone(),
                running_gamma: data[conv2_running_gamma].clone(),
                running_beta: data[conv2_running_beta].clone(),
                eps,
                steps: conv2_steps,
                padding: conv2_padding,
                dilation: conv2_dilation,
            },
            downsample: Some(DownSample {
                conv: Conv2dBatchNorm {
                    weight: data[downsample_weight]
                        .clone()
                        .permute([2, 3, 1, 0])
                        .expect("permute failed"),
                    bias: None,
                    running_mean: data[downsample_running_mean].clone(),
                    running_var: data[downsample_running_var].clone(),
                    running_gamma: data[downsample_running_gamma].clone(),
                    running_beta: data[downsample_running_beta].clone(),
                    eps,
                    steps: downsample_steps,
                    padding: downsample_padding,
                    dilation: downsample_dilation,
                },
            }),
        }
    }

    let layer1 = Sequential {
        layers: vec![
            create_basic_block(
                &data,
                "layer1_0_conv1_weight.txt",
                "layer1_0_bn1_running_mean.txt",
                "layer1_0_bn1_running_var.txt",
                "layer1_0_bn1_weight.txt",
                "layer1_0_bn1_bias.txt",
                "layer1_0_conv2_weight.txt",
                "layer1_0_bn2_running_mean.txt",
                "layer1_0_bn2_running_var.txt",
                "layer1_0_bn2_weight.txt",
                "layer1_0_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer1_1_conv1_weight.txt",
                "layer1_1_bn1_running_mean.txt",
                "layer1_1_bn1_running_var.txt",
                "layer1_1_bn1_weight.txt",
                "layer1_1_bn1_bias.txt",
                "layer1_1_conv2_weight.txt",
                "layer1_1_bn2_running_mean.txt",
                "layer1_1_bn2_running_var.txt",
                "layer1_1_bn2_weight.txt",
                "layer1_1_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer1_2_conv1_weight.txt",
                "layer1_2_bn1_running_mean.txt",
                "layer1_2_bn1_running_var.txt",
                "layer1_2_bn1_weight.txt",
                "layer1_2_bn1_bias.txt",
                "layer1_2_conv2_weight.txt",
                "layer1_2_bn2_running_mean.txt",
                "layer1_2_bn2_running_var.txt",
                "layer1_2_bn2_weight.txt",
                "layer1_2_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
        ],
    };

    let layer2 = Sequential {
        layers: vec![
            create_basic_block_with_downsample(
                &data,
                "layer2_0_conv1_weight.txt",
                "layer2_0_bn1_running_mean.txt",
                "layer2_0_bn1_running_var.txt",
                "layer2_0_bn1_weight.txt",
                "layer2_0_bn1_bias.txt",
                "layer2_0_conv2_weight.txt",
                "layer2_0_bn2_running_mean.txt",
                "layer2_0_bn2_running_var.txt",
                "layer2_0_bn2_weight.txt",
                "layer2_0_bn2_bias.txt",
                "layer2_0_downsample_0_weight.txt",
                "layer2_0_downsample_1_running_mean.txt",
                "layer2_0_downsample_1_running_var.txt",
                "layer2_0_downsample_1_weight.txt",
                "layer2_0_downsample_1_bias.txt",
                1e-5, /*eps */
                2,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
                2,    /*downsample_steps */
                0,    /*downsample_padding */
                1,    /*downsample_dilation */
            ),
            create_basic_block(
                &data,
                "layer2_1_conv1_weight.txt",
                "layer2_1_bn1_running_mean.txt",
                "layer2_1_bn1_running_var.txt",
                "layer2_1_bn1_weight.txt",
                "layer2_1_bn1_bias.txt",
                "layer2_1_conv2_weight.txt",
                "layer2_1_bn2_running_mean.txt",
                "layer2_1_bn2_running_var.txt",
                "layer2_1_bn2_weight.txt",
                "layer2_1_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer2_2_conv1_weight.txt",
                "layer2_2_bn1_running_mean.txt",
                "layer2_2_bn1_running_var.txt",
                "layer2_2_bn1_weight.txt",
                "layer2_2_bn1_bias.txt",
                "layer2_2_conv2_weight.txt",
                "layer2_2_bn2_running_mean.txt",
                "layer2_2_bn2_running_var.txt",
                "layer2_2_bn2_weight.txt",
                "layer2_2_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer2_3_conv1_weight.txt",
                "layer2_3_bn1_running_mean.txt",
                "layer2_3_bn1_running_var.txt",
                "layer2_3_bn1_weight.txt",
                "layer2_3_bn1_bias.txt",
                "layer2_3_conv2_weight.txt",
                "layer2_3_bn2_running_mean.txt",
                "layer2_3_bn2_running_var.txt",
                "layer2_3_bn2_weight.txt",
                "layer2_3_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
        ],
    };

    let layer3 = Sequential {
        layers: vec![
            create_basic_block_with_downsample(
                &data,
                "layer3_0_conv1_weight.txt",
                "layer3_0_bn1_running_mean.txt",
                "layer3_0_bn1_running_var.txt",
                "layer3_0_bn1_weight.txt",
                "layer3_0_bn1_bias.txt",
                "layer3_0_conv2_weight.txt",
                "layer3_0_bn2_running_mean.txt",
                "layer3_0_bn2_running_var.txt",
                "layer3_0_bn2_weight.txt",
                "layer3_0_bn2_bias.txt",
                "layer3_0_downsample_0_weight.txt",
                "layer3_0_downsample_1_running_mean.txt",
                "layer3_0_downsample_1_running_var.txt",
                "layer3_0_downsample_1_weight.txt",
                "layer3_0_downsample_1_bias.txt",
                1e-5, /*eps */
                2,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
                2,    /*downsample_steps */
                0,    /*downsample_padding */
                1,    /*downsample_dilation */
            ),
            create_basic_block(
                &data,
                "layer3_1_conv1_weight.txt",
                "layer3_1_bn1_running_mean.txt",
                "layer3_1_bn1_running_var.txt",
                "layer3_1_bn1_weight.txt",
                "layer3_1_bn1_bias.txt",
                "layer3_1_conv2_weight.txt",
                "layer3_1_bn2_running_mean.txt",
                "layer3_1_bn2_running_var.txt",
                "layer3_1_bn2_weight.txt",
                "layer3_1_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer3_2_conv1_weight.txt",
                "layer3_2_bn1_running_mean.txt",
                "layer3_2_bn1_running_var.txt",
                "layer3_2_bn1_weight.txt",
                "layer3_2_bn1_bias.txt",
                "layer3_2_conv2_weight.txt",
                "layer3_2_bn2_running_mean.txt",
                "layer3_2_bn2_running_var.txt",
                "layer3_2_bn2_weight.txt",
                "layer3_2_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer3_3_conv1_weight.txt",
                "layer3_3_bn1_running_mean.txt",
                "layer3_3_bn1_running_var.txt",
                "layer3_3_bn1_weight.txt",
                "layer3_3_bn1_bias.txt",
                "layer3_3_conv2_weight.txt",
                "layer3_3_bn2_running_mean.txt",
                "layer3_3_bn2_running_var.txt",
                "layer3_3_bn2_weight.txt",
                "layer3_3_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer3_4_conv1_weight.txt",
                "layer3_4_bn1_running_mean.txt",
                "layer3_4_bn1_running_var.txt",
                "layer3_4_bn1_weight.txt",
                "layer3_4_bn1_bias.txt",
                "layer3_4_conv2_weight.txt",
                "layer3_4_bn2_running_mean.txt",
                "layer3_4_bn2_running_var.txt",
                "layer3_4_bn2_weight.txt",
                "layer3_4_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
        ],
    };

    let layer4 = Sequential {
        layers: vec![
            create_basic_block_with_downsample(
                &data,
                "layer4_0_conv1_weight.txt",
                "layer4_0_bn1_running_mean.txt",
                "layer4_0_bn1_running_var.txt",
                "layer4_0_bn1_weight.txt",
                "layer4_0_bn1_bias.txt",
                "layer4_0_conv2_weight.txt",
                "layer4_0_bn2_running_mean.txt",
                "layer4_0_bn2_running_var.txt",
                "layer4_0_bn2_weight.txt",
                "layer4_0_bn2_bias.txt",
                "layer4_0_downsample_0_weight.txt",
                "layer4_0_downsample_1_running_mean.txt",
                "layer4_0_downsample_1_running_var.txt",
                "layer4_0_downsample_1_weight.txt",
                "layer4_0_downsample_1_bias.txt",
                1e-5, /*eps */
                2,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
                2,    /*downsample_steps */
                0,    /*downsample_padding */
                1,    /*downsample_dilation */
            ),
            create_basic_block(
                &data,
                "layer4_1_conv1_weight.txt",
                "layer4_1_bn1_running_mean.txt",
                "layer4_1_bn1_running_var.txt",
                "layer4_1_bn1_weight.txt",
                "layer4_1_bn1_bias.txt",
                "layer4_1_conv2_weight.txt",
                "layer4_1_bn2_running_mean.txt",
                "layer4_1_bn2_running_var.txt",
                "layer4_1_bn2_weight.txt",
                "layer4_1_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
            create_basic_block(
                &data,
                "layer4_2_conv1_weight.txt",
                "layer4_2_bn1_running_mean.txt",
                "layer4_2_bn1_running_var.txt",
                "layer4_2_bn1_weight.txt",
                "layer4_2_bn1_bias.txt",
                "layer4_2_conv2_weight.txt",
                "layer4_2_bn2_running_mean.txt",
                "layer4_2_bn2_running_var.txt",
                "layer4_2_bn2_weight.txt",
                "layer4_2_bn2_bias.txt",
                1e-5, /*eps */
                1,    /*conv1_steps */
                1,    /*conv1_padding */
                1,    /*conv1_dilation */
                1,    /*conv2_steps */
                1,    /*conv2_padding */
                1,    /*conv2_dilation */
            ),
        ],
    };

    let avg_pool = AdaptiveAvgPool2d {
        kernel_size: [1, 1],
    };

    let fc = Linear {
        weight: data["fc_weight.txt"].clone(),
        bias: data["fc_bias.txt"].clone(),
    };

    ResNet {
        conv1: bn_conv1,
        max_pool1,
        layer1,
        layer2,
        layer3,
        layer4,
        avg_pool,
        fc,
    }
}

fn main() -> anyhow::Result<()> {
    // let refresh_kind = RefreshKind::default().with_memory(MemoryRefreshKind::everything());
    // let mut sys = System::new_with_specifics(refresh_kind);
    // sys.refresh_all();
    // let pid = sysinfo::get_current_pid().expect("Failed to get current PID");
    // sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    // if let Some(process) = sys.process(pid) {
    //     println!("Before Inference - Memory usage: {} KB", process.memory());
    // }
    // let resnet = create_resnet();
    // resnet.save("resnet.model")?;
    let data = ResNet::load("resnet.model")?;
    let input = Tensor::<f32>::randn(&[5, 128, 128, 3])?;
    let now = std::time::Instant::now();
    // for _ in 0..10 {
        let output = data.forward(&input)?;
    // }
    println!("time: {:?}", now.elapsed() / 10);
    // sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), false);
    // if let Some(process) = sys.process(pid) {
    //     println!("After Inference - Memory usage: {} KB", process.memory());
    // }
    Ok(())
}

