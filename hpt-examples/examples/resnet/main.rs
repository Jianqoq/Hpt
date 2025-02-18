
use hpt::*;
use safetensors::SafeTensors;

type F32Simd = <f32 as TypeCommon>::Vec;

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
        let out = self.bn_conv1.forward(&x, |x| x._relu())?;
        let mut out = self.bn_conv2.forward(&out, |x| x)?;
        out.par_iter_mut_simd()
            .zip(identity.par_iter_simd())
            .for_each(
                |(a, b)| *a = (*a + b)._relu(),
                |(a, b)| {
                    a.write_unaligned((a.read_unaligned() + b)._relu());
                },
            );
        Ok(out)
    }
}

impl Linear {
    pub fn forward(&self, x: &Tensor<f32>) -> anyhow::Result<Tensor<f32>> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out.add_(&self.bias, out.clone())?)
    }
}

#[derive(Save, Load)]
pub struct Conv2dBatchNorm {
    weight_str: String,
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
    bn_conv1: Conv2dBatchNorm,
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
        let x = self.bn_conv1.forward(&x, |x| x._relu())?;
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
fn create_resnet() -> anyhow::Result<ResNet> {
    let data = std::fs::read("resnet.safetensor").expect("failed to read weights");
    let data = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize weights");
    let max_pool1 = MaxPool2d {
        kernel_size: Shape::new([3, 3]),
        stride: 2,
        padding: 1,
        dilation: 1,
    };

    fn create_basic_block(
        data: &SafeTensors,
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
                weight_str: weight.to_string(),
                weight: Tensor::<f32>::from_safe_tensors(data, weight)
                    .permute([2, 3, 1, 0])
                    .expect("permute failed")
                    .contiguous()
                    .expect("contiguous failed"),
                bias: None,
                running_mean: Tensor::<f32>::from_safe_tensors(data, running_mean),
                running_var: Tensor::<f32>::from_safe_tensors(data, running_var),
                running_gamma: Tensor::<f32>::from_safe_tensors(data, running_gamma),
                running_beta: Tensor::<f32>::from_safe_tensors(data, running_beta),
                eps,
                steps: conv1_steps,
                padding: conv1_padding,
                dilation: conv1_dilation,
            },
            bn_conv2: Conv2dBatchNorm {
                weight_str: conv2_weight.to_string(),
                weight: Tensor::<f32>::from_safe_tensors(data, conv2_weight)
                    .permute([2, 3, 1, 0])
                    .expect("permute failed")
                    .contiguous()
                    .expect("contiguous failed"),
                bias: None,
                running_mean: Tensor::<f32>::from_safe_tensors(data, conv2_running_mean),
                running_var: Tensor::<f32>::from_safe_tensors(data, conv2_running_var),
                running_gamma: Tensor::<f32>::from_safe_tensors(data, conv2_running_gamma),
                running_beta: Tensor::<f32>::from_safe_tensors(data, conv2_running_beta),
                eps,
                steps: conv2_steps,
                padding: conv2_padding,
                dilation: conv2_dilation,
            },
            downsample: None,
        }
    }

    fn create_basic_block_with_downsample(
        data: &SafeTensors,
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
                weight_str: weight.to_string(),
                weight: Tensor::<f32>::from_safe_tensors(data, weight)
                    .permute([2, 3, 1, 0])
                    .expect("permute failed")
                    .contiguous()
                    .expect("contiguous failed"),
                bias: None,
                running_mean: Tensor::<f32>::from_safe_tensors(data, running_mean),
                running_var: Tensor::<f32>::from_safe_tensors(data, running_var),
                running_gamma: Tensor::<f32>::from_safe_tensors(data, running_gamma),
                running_beta: Tensor::<f32>::from_safe_tensors(data, running_beta),
                eps,
                steps: conv1_steps,
                padding: conv1_padding,
                dilation: conv1_dilation,
            },
            bn_conv2: Conv2dBatchNorm {
                weight_str: conv2_weight.to_string(),
                weight: Tensor::<f32>::from_safe_tensors(data, conv2_weight)
                    .permute([2, 3, 1, 0])
                    .expect("permute failed")
                    .contiguous()
                    .expect("contiguous failed"),
                bias: None,
                running_mean: Tensor::<f32>::from_safe_tensors(data, conv2_running_mean),
                running_var: Tensor::<f32>::from_safe_tensors(data, conv2_running_var),
                running_gamma: Tensor::<f32>::from_safe_tensors(data, conv2_running_gamma),
                running_beta: Tensor::<f32>::from_safe_tensors(data, conv2_running_beta),
                eps,
                steps: conv2_steps,
                padding: conv2_padding,
                dilation: conv2_dilation,
            },
            downsample: Some(DownSample {
                conv: Conv2dBatchNorm {
                    weight_str: downsample_weight.to_string(),
                    weight: Tensor::<f32>::from_safe_tensors(data, downsample_weight)
                        .permute([2, 3, 1, 0])
                        .expect("permute failed")
                        .contiguous()
                        .expect("contiguous failed"),
                    bias: None,
                    running_mean: Tensor::<f32>::from_safe_tensors(data, downsample_running_mean),
                    running_var: Tensor::<f32>::from_safe_tensors(data, downsample_running_var),
                    running_gamma: Tensor::<f32>::from_safe_tensors(data, downsample_running_gamma),
                    running_beta: Tensor::<f32>::from_safe_tensors(data, downsample_running_beta),
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
                "layer1.0.conv1.weight",
                "layer1.0.bn1.running_mean",
                "layer1.0.bn1.running_var",
                "layer1.0.bn1.weight",
                "layer1.0.bn1.bias",
                "layer1.0.conv2.weight",
                "layer1.0.bn2.running_mean",
                "layer1.0.bn2.running_var",
                "layer1.0.bn2.weight",
                "layer1.0.bn2.bias",
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
                "layer1.1.conv1.weight",
                "layer1.1.bn1.running_mean",
                "layer1.1.bn1.running_var",
                "layer1.1.bn1.weight",
                "layer1.1.bn1.bias",
                "layer1.1.conv2.weight",
                "layer1.1.bn2.running_mean",
                "layer1.1.bn2.running_var",
                "layer1.1.bn2.weight",
                "layer1.1.bn2.bias",
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
                "layer1.2.conv1.weight",
                "layer1.2.bn1.running_mean",
                "layer1.2.bn1.running_var",
                "layer1.2.bn1.weight",
                "layer1.2.bn1.bias",
                "layer1.2.conv2.weight",
                "layer1.2.bn2.running_mean",
                "layer1.2.bn2.running_var",
                "layer1.2.bn2.weight",
                "layer1.2.bn2.bias",
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
                "layer2.0.conv1.weight",
                "layer2.0.bn1.running_mean",
                "layer2.0.bn1.running_var",
                "layer2.0.bn1.weight",
                "layer2.0.bn1.bias",
                "layer2.0.conv2.weight",
                "layer2.0.bn2.running_mean",
                "layer2.0.bn2.running_var",
                "layer2.0.bn2.weight",
                "layer2.0.bn2.bias",
                "layer2.0.downsample.0.weight",
                "layer2.0.downsample.1.running_mean",
                "layer2.0.downsample.1.running_var",
                "layer2.0.downsample.1.weight",
                "layer2.0.downsample.1.bias",
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
                "layer2.1.conv1.weight",
                "layer2.1.bn1.running_mean",
                "layer2.1.bn1.running_var",
                "layer2.1.bn1.weight",
                "layer2.1.bn1.bias",
                "layer2.1.conv2.weight",
                "layer2.1.bn2.running_mean",
                "layer2.1.bn2.running_var",
                "layer2.1.bn2.weight",
                "layer2.1.bn2.bias",
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
                "layer2.2.conv1.weight",
                "layer2.2.bn1.running_mean",
                "layer2.2.bn1.running_var",
                "layer2.2.bn1.weight",
                "layer2.2.bn1.bias",
                "layer2.2.conv2.weight",
                "layer2.2.bn2.running_mean",
                "layer2.2.bn2.running_var",
                "layer2.2.bn2.weight",
                "layer2.2.bn2.bias",
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
                "layer2.3.conv1.weight",
                "layer2.3.bn1.running_mean",
                "layer2.3.bn1.running_var",
                "layer2.3.bn1.weight",
                "layer2.3.bn1.bias",
                "layer2.3.conv2.weight",
                "layer2.3.bn2.running_mean",
                "layer2.3.bn2.running_var",
                "layer2.3.bn2.weight",
                "layer2.3.bn2.bias",
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
                "layer3.0.conv1.weight",
                "layer3.0.bn1.running_mean",
                "layer3.0.bn1.running_var",
                "layer3.0.bn1.weight",
                "layer3.0.bn1.bias",
                "layer3.0.conv2.weight",
                "layer3.0.bn2.running_mean",
                "layer3.0.bn2.running_var",
                "layer3.0.bn2.weight",
                "layer3.0.bn2.bias",
                "layer3.0.downsample.0.weight",
                "layer3.0.downsample.1.running_mean",
                "layer3.0.downsample.1.running_var",
                "layer3.0.downsample.1.weight",
                "layer3.0.downsample.1.bias",
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
                "layer3.1.conv1.weight",
                "layer3.1.bn1.running_mean",
                "layer3.1.bn1.running_var",
                "layer3.1.bn1.weight",
                "layer3.1.bn1.bias",
                "layer3.1.conv2.weight",
                "layer3.1.bn2.running_mean",
                "layer3.1.bn2.running_var",
                "layer3.1.bn2.weight",
                "layer3.1.bn2.bias",
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
                "layer3.2.conv1.weight",
                "layer3.2.bn1.running_mean",
                "layer3.2.bn1.running_var",
                "layer3.2.bn1.weight",
                "layer3.2.bn1.bias",
                "layer3.2.conv2.weight",
                "layer3.2.bn2.running_mean",
                "layer3.2.bn2.running_var",
                "layer3.2.bn2.weight",
                "layer3.2.bn2.bias",
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
                "layer3.3.conv1.weight",
                "layer3.3.bn1.running_mean",
                "layer3.3.bn1.running_var",
                "layer3.3.bn1.weight",
                "layer3.3.bn1.bias",
                "layer3.3.conv2.weight",
                "layer3.3.bn2.running_mean",
                "layer3.3.bn2.running_var",
                "layer3.3.bn2.weight",
                "layer3.3.bn2.bias",
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
                "layer3.4.conv1.weight",
                "layer3.4.bn1.running_mean",
                "layer3.4.bn1.running_var",
                "layer3.4.bn1.weight",
                "layer3.4.bn1.bias",
                "layer3.4.conv2.weight",
                "layer3.4.bn2.running_mean",
                "layer3.4.bn2.running_var",
                "layer3.4.bn2.weight",
                "layer3.4.bn2.bias",
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
                "layer3.5.conv1.weight",
                "layer3.5.bn1.running_mean",
                "layer3.5.bn1.running_var",
                "layer3.5.bn1.weight",
                "layer3.5.bn1.bias",
                "layer3.5.conv2.weight",
                "layer3.5.bn2.running_mean",
                "layer3.5.bn2.running_var",
                "layer3.5.bn2.weight",
                "layer3.5.bn2.bias",
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
                "layer4.0.conv1.weight",
                "layer4.0.bn1.running_mean",
                "layer4.0.bn1.running_var",
                "layer4.0.bn1.weight",
                "layer4.0.bn1.bias",
                "layer4.0.conv2.weight",
                "layer4.0.bn2.running_mean",
                "layer4.0.bn2.running_var",
                "layer4.0.bn2.weight",
                "layer4.0.bn2.bias",
                "layer4.0.downsample.0.weight",
                "layer4.0.downsample.1.running_mean",
                "layer4.0.downsample.1.running_var",
                "layer4.0.downsample.1.weight",
                "layer4.0.downsample.1.bias",
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
                "layer4.1.conv1.weight",
                "layer4.1.bn1.running_mean",
                "layer4.1.bn1.running_var",
                "layer4.1.bn1.weight",
                "layer4.1.bn1.bias",
                "layer4.1.conv2.weight",
                "layer4.1.bn2.running_mean",
                "layer4.1.bn2.running_var",
                "layer4.1.bn2.weight",
                "layer4.1.bn2.bias",
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
                "layer4.2.conv1.weight",
                "layer4.2.bn1.running_mean",
                "layer4.2.bn1.running_var",
                "layer4.2.bn1.weight",
                "layer4.2.bn1.bias",
                "layer4.2.conv2.weight",
                "layer4.2.bn2.running_mean",
                "layer4.2.bn2.running_var",
                "layer4.2.bn2.weight",
                "layer4.2.bn2.bias",
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
        weight: Tensor::<f32>::from_safe_tensors(&data, "fc.weight"),
        bias: Tensor::<f32>::from_safe_tensors(&data, "fc.bias"),
    };
    Ok(ResNet {
        bn_conv1: Conv2dBatchNorm {
            weight_str: "conv1.weight".to_string(),
            weight: Tensor::<f32>::from_safe_tensors(&data, "conv1.weight")
                .permute(&[2, 3, 1, 0])?
                .contiguous()?,
            bias: None,
            running_mean: Tensor::<f32>::from_safe_tensors(&data, "bn1.running_mean"),
            running_var: Tensor::<f32>::from_safe_tensors(&data, "bn1.running_var"),
            running_gamma: Tensor::<f32>::from_safe_tensors(&data, "bn1.weight"),
            running_beta: Tensor::<f32>::from_safe_tensors(&data, "bn1.bias"),
            eps: 1e-5,
            steps: 2,
            padding: 3,
            dilation: 1,
        },
        max_pool1,
        layer1,
        layer2,
        layer3,
        layer4,
        avg_pool,
        fc,
    })
}
fn main() -> anyhow::Result<()> {
    let resnet = create_resnet()?;
    // you can save the model to a file
    resnet.save("resnet.model")?;
    // you can load the model from a file
    let resnet = ResNet::load("resnet.model")?;
    let mut size = vec![];
    let mut time = vec![];
    for i in 0..50 {
        let inp = Tensor::<f32>::randn([5, 64 + 32 * i, 64 + 32 * i, 3])?;
        let now = std::time::Instant::now();
        for _ in 0..10 {
            resnet.forward(&inp)?;
        }
        size.push(64 + 32 * i);
        time.push((now.elapsed() / 10).as_secs_f32() * 1000.0);
        println!("size: {:?}, time: {:?}", size.last().unwrap(), time.last().unwrap());
    }
    println!("size: {:?}", size);
    println!("time: {:?}", time);
    Ok(())
}
