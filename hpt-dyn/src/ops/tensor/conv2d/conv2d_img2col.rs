use hpt_common::error::base::TensorError;
use hpt_common::layout::layout::Layout;
use hpt_common::shape::shape::Shape;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;

use crate::ops::tensor::conv2d::utils::{ cal_conv2d_output_shape, handle_post };
use crate::ops::tensor::matmul::matmul::matmul_with_out;
use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;
use crate::Tensor;

use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::create_packed_input_img2col;
use super::utils::img2col_nhwc;

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel>(
    input: &Tensor,
    kernels: &Tensor,
    bias: Option<&Tensor>,
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
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(T::Vec) -> T::Vec>,
    output: Tensor
) -> Result<Tensor, TensorError>
    where T: MatmulMicroKernel
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
        &[
            (ph_start, ph_end),
            (pw_start, pw_end),
        ],
        &[step_height, step_width],
        &[dh, dw]
    );
    let img = input.clone();
    let inp_ptr = input.ptr();
    let (input_buffer, input_buffer_layout) = create_packed_input_img2col::<T>(
        batch,
        kh,
        kw,
        in_channels,
        out_height,
        out_width
    );

    assert_eq!(output.shape().as_slice(), &[batch, out_height, out_width, out_channels]);
    img2col_nhwc(
        input_buffer,
        inp_ptr,
        batch,
        &img.strides(),
        img_height,
        img_width,
        in_channels,
        out_height,
        out_width,
        [kh, kw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw]
    );

    let buffer_tensor = (unsafe {
        Tensor::from_raw(
            input_buffer.ptr as *mut _,
            Layout::from(Shape::new([batch, out_height * out_width, kh * kw * in_channels])),
            input.dtype,
            input.device.clone()
        )
    })?;

    let output_shape = output.shape().clone();
    let mut res = matmul_with_out(
        &buffer_tensor,
        &kernels.reshape(&[kh * kw * in_channels, out_channels])?,
        Some(output),
        None::<fn(T, usize, usize) -> T>,
        None::<fn(T::Vec, usize, usize) -> T::Vec>
    )?.reshape(&output_shape)?;

    handle_post(&mut res, bias, post_scalar, post_vec)?;

    unsafe {
        std::alloc::dealloc(input_buffer.ptr as *mut _, input_buffer_layout);
    }
    Ok(res)
}
