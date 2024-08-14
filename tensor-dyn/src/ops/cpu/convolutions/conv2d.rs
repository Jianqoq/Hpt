use num::traits::MulAdd;
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_traits::TensorCreator;
use tensor_types::type_promote::NormalOut;
use crate::tensor_base::_Tensor;

pub fn conv2d<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>
{
    let img_shape = img.shape();
    let img_channels = img_shape[0];
    let img_width = img_shape[1];
    let img_height = img_shape[2];
    let kernel_shape = kernels.shape();
    let in_channels = kernel_shape[0];
    let out_channels = kernel_shape[1];
    let kernel_width = kernel_shape[2];
    let kernel_height = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_channels, out_width, out_height])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();
    let os0 = output.strides()[0];
    let os1 = output.strides()[1];
    let os2 = output.strides()[2];

    let is0 = img.strides()[0];
    let is1 = img.strides()[1];
    let is2 = img.strides()[2];

    let ks0 = kernels.strides()[0];
    let ks1 = kernels.strides()[1];
    let ks2 = kernels.strides()[2];
    let ks3 = kernels.strides()[3];

    for l in 0..out_height {
        for n in 0..kernel_height {
            for m in 0..kernel_width {
                for i in 0..in_channels {
                    for k in 0..out_width {
                        for j in 0..out_channels {
                            let k_val = kernel[i * ks0 + j * ks1 + m * ks2 + n * ks3];
                            let i_val = inp[i * is0 + (k * step_width + m) * is1 + (l * step_height + n) * is2]; // prettier-ignore
                            out[j * os0 + k * os1 + l * os2] = i_val.mul_add(
                                k_val,
                                out[j * os0 + k * os1 + l * os2]
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub fn conv2d_naive<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>
{
    let img_shape = img.shape();
    let img_channels = img_shape[0];
    let img_width = img_shape[1];
    let img_height = img_shape[2];
    let kernel_shape = kernels.shape();
    let out_channels = kernel_shape[0];
    let in_channels = kernel_shape[1];
    let kernel_width = kernel_shape[2];
    let kernel_height = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_channels, out_width, out_height])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();
    let os0 = output.strides()[0];
    let os1 = output.strides()[1];
    let os2 = output.strides()[2];

    let is0 = img.strides()[0];
    let is1 = img.strides()[1];
    let is2 = img.strides()[2];

    let ks0 = kernels.strides()[0];
    let ks1 = kernels.strides()[1];
    let ks2 = kernels.strides()[2];
    let ks3 = kernels.strides()[3];

    for j in 0..out_channels {
        for i in 0..in_channels {
            for l in 0..out_height {
                for k in 0..out_width {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            let k_val = kernel[j * ks0 + i * ks1 + n * ks2 + m * ks3];
                            let i_val = inp[i * is0 + (l * step_height + n) * is1 + (k * step_width + m) * is2]; // prettier-ignore
                            out[j * os0 + l * os1 + k * os2] = i_val.mul_add(
                                k_val,
                                out[j * os0 + l * os1 + k * os2]
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}
