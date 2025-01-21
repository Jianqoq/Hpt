use tensor_dyn::{ traits::SimdMath, Load, Save, Shape, Tensor, TensorError, TypeCommon };
use tensor_dyn::{
    NormalOutUnary, ParStridedIteratorSimd, ParStridedIteratorSimdZip, Random, TensorIterator
};
use tensor_dyn::Matmul;
use tensor_dyn::ShapeManipulate;
use tensor_dyn::NormalBinOps;

type F32Simd = <f32 as TypeCommon>::Vec;

impl Conv2dBatchNorm {
    #[track_caller]
    pub fn forward(
        &self,
        x: &Tensor<f32>,
        activation: fn(F32Simd) -> F32Simd
    ) -> Result<Tensor<f32>, TensorError> {
        Ok(
            x.batchnorm_conv2d(
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
                Some(activation)
            )?
        )
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
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        Ok(
            x.maxpool2d(
                &self.kernel_size,
                [self.stride as i64, self.stride as i64],
                [
                    (self.padding as i64, self.padding as i64),
                    (self.padding as i64, self.padding as i64),
                ],
                [self.dilation as i64, self.dilation as i64]
            )?
        )
    }
}

#[derive(Save, Load)]
pub struct AdaptiveAvgPool2d {
    kernel_size: [i64; 2],
}

impl AdaptiveAvgPool2d {
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        Ok(x.adaptive_avgpool2d(self.kernel_size)?)
    }
}

#[derive(Save, Load)]
pub struct DownSample {
    conv: Conv2dBatchNorm,
}

impl DownSample {
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        Ok(self.conv.forward(x, |x| x)?)
    }
}

#[derive(Save, Load)]
pub struct Sequential {
    layers: Vec<BasicBlock>,
}

impl Sequential {
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
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
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
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
                |(a, b)| {
                    *a = (*a + b)._relu();
                },
                |(a, b)| {
                    a.write_unaligned((a.read_unaligned() + b).relu());
                }
            );
        Ok(out)
    }
}

impl Linear {
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let out = x.matmul(&self.weight.t()?)?;
        Ok(out.add_(&self.bias, &out)?)
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
    pub fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        let x = self.bn_conv1.forward(&x, |x| x.relu())?;
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

fn main() -> Result<(), TensorError> {
    let resnet = ResNet::load("resnet.model").unwrap();
    let input = Tensor::<f32>::randn([1, 3, 224, 224])?;
    let output = resnet.forward(&input)?;
    println!("{:?}", output);
    Ok(())
}
