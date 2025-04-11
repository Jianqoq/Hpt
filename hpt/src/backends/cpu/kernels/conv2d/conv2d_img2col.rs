use crate::backends::common::conv::cal_conv2d_output_shape;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_common::layout::layout::Layout;
use hpt_common::shape::shape::Shape;
use hpt_traits::ops::binary::Gemm;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::shape_manipulate::ShapeManipulate;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::into_scalar::Cast;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::pack_kernel;
use super::utils::{calculate_kernel_params, create_packed_input_img2col};
use super::utils::{create_packed_kernel, img2col_nhwc};

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    batch: i64,
    img_height: i64,
    img_width: i64,
    img_channels: i64,
    out_channels: i64,
    kh: i64,
    kw: i64,
    output: _Tensor<T, Cpu, DEVICE, A>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let in_channels = img_channels;
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
    let img = input.clone();
    let inp_ptr = input.ptr();
    let (input_buffer, input_buffer_layout) =
        create_packed_input_img2col::<T>(kh, kw, in_channels, out_height, out_width);

    assert_eq!(output.shape().as_slice(), &[batch, out_height, out_width, out_channels]);
    img2col_nhwc(
        input_buffer,
        inp_ptr,
        &img.strides(),
        img_height,
        img_width,
        in_channels,
        out_height,
        out_width,
        [kh, kw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
    );

    let buffer_tensor = unsafe {
        _Tensor::<T, Cpu, DEVICE, A>::from_raw(
            input_buffer.ptr as *mut _,
            Shape::new([out_height * out_width, kh * kw * in_channels]),
        )
    }?;

    let output_shape = output.shape().clone();
    let res = buffer_tensor.gemm_(
        &kernels.reshape(&[kh * kw * in_channels, out_channels])?,
        T::ZERO,
        T::ONE,
        false,
        false,
        false,
        output,
    )?.reshape(output_shape)?;

    unsafe {
        std::alloc::dealloc(input_buffer.ptr as *mut _, input_buffer_layout);
    }
    Ok(res)
}
