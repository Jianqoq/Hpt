#![feature(portable_simd)]

use num::traits::MulAdd;
use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_traits::TensorCreator;
use tensor_types::type_promote::NormalOut;
use crate::tensor_base::_Tensor;

/// image: `[height, width, channels]`
///
/// kernels: `[kernel_height, kernel_width, in_channels, out_channels]`
///
/// steps: `[step_width, step_height]`
///
/// output: `[out_width, out_height, out_channels]`
#[cfg(target_feature = "fma")]
pub fn conv2d<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>
{
    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    for l in 0..out_height {
        for n in 0..kernel_height {
            for m in 0..kernel_width {
                for i in 0..in_channels {
                    for k in 0..out_width {
                        for j in 0..out_channels {
                            let k_val = kernel[i * ks2 + j * ks3 + m * ks1 + n * ks0];
                            let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                            out[j * os2 + k * os1 + l * os0] += i_val * k_val;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

#[cfg(target_feature = "fma")]
pub fn conv2d_block<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>
{
    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let c_ob = 4;
    let c_ib = 4;
    let w_ob = 5;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let ip_end = (in_channels + c_ib - 1) / c_ib;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    for jp in 0..jp_end {
        for ip in 0..ip_end {
            for l in 0..out_height {
                for kp in 0..kp_end {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for i in 0..c_ib {
                                let i = ip * c_ib + i;
                                for k in 0..w_ob {
                                    let k = kp * w_ob + k;
                                    for j in 0..c_ob {
                                        let j = jp * c_ob + j;
                                        let k_val = kernel[i * ks2 + j * ks3 + m * ks1 + n * ks0];
                                        let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                        out[j * os2 + k * os1 + l * os0] += i_val * k_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

use tensor_types::into_scalar::IntoScalar;
#[cfg(target_feature = "fma")]
pub fn conv2d_block_simd<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
)
    -> anyhow::Result<_Tensor<T>>
    where
        T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>,
        T: IntoScalar<f32>
{
    use wide::{ f32x4, f32x8 };

    let img_shape = img.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_channels = img_shape[2];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_height, out_width, out_channels])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();

    let os0 = output.strides()[0]; // height
    let os1 = output.strides()[1]; // width
    let os2 = output.strides()[2]; // channels

    let is0 = img.strides()[0]; // height
    let is1 = img.strides()[1]; // width
    let is2 = img.strides()[2]; // channels

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels
    let ks3 = kernels.strides()[3]; // out_channels

    let c_ob = 8;
    let c_ib = 4;
    let w_ob = 5;
    let jp_end = (out_channels + c_ob - 1) / c_ob;
    let ip_end = (in_channels + c_ib - 1) / c_ib;
    let kp_end = (out_width + w_ob - 1) / w_ob;
    for jp in 0..jp_end {
        for ip in 0..ip_end {
            for l in 0..out_height {
                for kp in 0..kp_end {
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            for i in 0..c_ib {
                                let i = ip * c_ib + i;
                                for k in 0..w_ob {
                                    let k = kp * w_ob + k;

                                    let kernel_ptr = &kernel[i * ks2 + jp * c_ob * ks3 + m * ks1 + n * ks0] as *const T; // prettier-ignore
                                    let kernel_vec = unsafe { std::slice::from_raw_parts(kernel_ptr, 8) }; // prettier-ignore
                                    let kernel_vector = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(kernel_vec)) }; // prettier-ignore

                                    let res_ptr = &mut out[jp * c_ob * os2 + k * os1 + l * os0]; // prettier-ignore
                                    let res_vec = unsafe { std::slice::from_raw_parts_mut(res_ptr, 8) }; // prettier-ignore
                                    let res_vector = unsafe { f32x8::from(std::mem::transmute::<&[T], &[f32]>(res_vec)) }; // prettier-ignore

                                    let i_val = inp[i * is2 + (k * step_width + m) * is1 + (l * step_height + n) * is0]; // prettier-ignore
                                    let scalar_vec = f32x8::new([i_val.into_scalar(); 8]);
                                    let res = kernel_vector.mul_add(scalar_vec, res_vector); // prettier-ignore

                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            res.to_array().as_ptr() as *const f32,
                                            res_vec.as_mut_ptr() as *mut f32,
                                            8
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// img: `[channels, height, width]`
///
/// kernels: `[out_channels, in_channels, kernel_height, kernel_width]`
///
/// steps: `[step_width, step_height]`
///
/// output: `[out_channels, out_height, out_width]`
pub fn conv2d_naive<T>(
    img: &_Tensor<T>,
    kernels: &_Tensor<T>,
    steps: [i64; 2]
) -> anyhow::Result<_Tensor<T>>
    where T: CommonBounds + std::ops::Mul<Output = T> + std::ops::AddAssign<T> + MulAdd<Output = T>
{
    let img_shape = img.shape();
    let img_channels = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let kernel_shape = kernels.shape();
    let out_channels = kernel_shape[0];
    let in_channels = kernel_shape[1];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];
    if in_channels != img_channels {
        panic!(
            "The number of input channels in the image must be equal to the number of input channels in the kernel."
        );
    }
    let (step_width, step_height) = (steps[0], steps[1]);

    let out_height =
        <i64 as NormalOut<i64>>::_floor((img_height - kernel_height) / step_height) + 1;
    let out_width = <i64 as NormalOut<i64>>::_floor((img_width - kernel_width) / step_width) + 1;
    let output = _Tensor::<T>::zeros([out_channels, out_height, out_width])?;
    let mut out = output.ptr();
    let inp = img.ptr();
    let kernel = kernels.ptr();
    let os0 = output.strides()[0]; // out_channels
    let os1 = output.strides()[1]; // out_height
    let os2 = output.strides()[2]; // out_width

    let is0 = img.strides()[0]; // channels
    let is1 = img.strides()[1]; // height
    let is2 = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // out_channels
    let ks1 = kernels.strides()[1]; // in_channels
    let ks2 = kernels.strides()[2]; // kernel_height
    let ks3 = kernels.strides()[3]; // kernel_width

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
