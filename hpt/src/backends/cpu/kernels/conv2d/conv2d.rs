use crate::backends::common::conv::cal_conv2d_output_shape;
use crate::backends::cpu::kernels::matmul::microkernel_trait::MatmulMicroKernel;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;

use super::microkernel_trait::Conv2dMicroKernel;
use super::{conv2d_direct, conv2d_img2col};

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: MatmulMicroKernel,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    ShapeError::check_contiguous(
        "Conv2d requires input tensor to be contiguous. ".to_string(),
        input.layout(),
    )?;
    ShapeError::check_contiguous(
        "Conv2d requires kernel tensor to be contiguous. ".to_string(),
        kernels.layout(),
    )?;
    if bias.is_some() {
        ShapeError::check_contiguous(
            "Conv2d requires bias tensor to be contiguous. ".to_string(),
            bias.unwrap().layout(),
        )?;
    }
    let img_shape = input.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
    let kernel_shape = kernels.shape();
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        return Err((ShapeError::ConvError {
            message: format!(
                "kernel in_channel {} not match input in_channel {}",
                in_channels, img_channels
            ),
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let (out_height, out_width) = cal_conv2d_output_shape(
        img_height,
        img_width,
        kh,
        kw,
        &[(ph_start, ph_end), (pw_start, pw_end)],
        &[step_height, step_width],
        &[dh, dw],
    );
    if out_height <= 0 || out_width <= 0 {
        return Err((ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let output = _Tensor::<T, Cpu, DEVICE, A>::empty([batch, out_height, out_width, out_channels])?;
    let img2col_buffer_size = kh * kw * in_channels * out_height * out_width;
    let direct_buffer_size = kh * kw * in_channels * out_channels;
    if img2col_buffer_size < direct_buffer_size {
        conv2d_img2col::conv2d(
            input,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            batch,
            img_height,
            img_width,
            img_channels,
            out_channels,
            kh,
            kw,
            post_scalar,
            post_vec,
            output,
        )
    } else {
        conv2d_direct::conv2d(
            input,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            batch,
            img_height,
            img_width,
            img_channels,
            out_channels,
            kh,
            kw,
            post_scalar,
            post_vec,
            output,
        )
    }
}
