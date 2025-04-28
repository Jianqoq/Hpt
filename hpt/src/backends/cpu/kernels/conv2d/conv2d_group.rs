use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::create_packed_kernel;
use super::utils::pack_kernel;
use super::utils::{calculate_kernel_params, handle_post};

pub(crate) fn conv2d_group<T, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    groups: i64,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
where
    T: Conv2dMicroKernel + CommonBounds,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    let img_shape = input.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let in_channels = img_shape[3];
    let kernel_shape = kernels.shape();
    let kernel_height = kernel_shape[0];
    let kernel_width = kernel_shape[1];
    let k_in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels / groups != k_in_channels {
        panic!(
            "kernel in_channel must equal to in_channel / groups, got {} and {}",
            k_in_channels,
            in_channels / groups
        );
    }
    if in_channels % groups != 0 {
        panic!("The number of input channels must be divisible by the number of groups.");
    }
    if out_channels % groups != 0 {
        panic!("The number of output channels must be divisible by the number of groups.");
    }
    let in_channels_per_group = in_channels / groups;
    let out_channels_per_group = out_channels / groups;
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height =
        (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
    let out_width = (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
    let img = input.clone();
    if out_height <= 0 || out_width <= 0 {
        return Err(ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let mut output =
        _Tensor::<T, Cpu, DEVICE, A>::empty([batch, out_height, out_width, out_channels])?;
    let out = output.ptr::<T>();

    let osb = output.strides()[0]; // batch
    let osh = output.strides()[1]; // height
    let osw = output.strides()[2]; // width

    let isb = img.strides()[0]; // batch
    let ish = img.strides()[1]; // height
    let isw = img.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let outer = batch * out_height;

    let inp_ptr = input.ptr::<T>();
    let kernel_ptr = kernels.ptr::<T>();
    let nr = T::get_max_nr() * T::Vec::SIZE;
    let mr = T::get_max_mr().min(out_width as usize);
    let param = calculate_kernel_params::<T>(
        in_channels,
        out_channels,
        out_width,
        mr,
        nr,
        [kernel_height as usize, kernel_width as usize],
    );
    let oc: i64 = param.nc as i64;
    let ic: i64 = param.kc as i64;
    let kc: i64 = param.mc as i64;
    let buffer = create_packed_kernel::<T, DEVICE, A>(
        kernel_height,
        kernel_width,
        in_channels_per_group,
        out_channels_per_group,
        oc,
        nr as i64,
    )?;

    let need_pad = ph_start != 0 || pw_start != 0 || ph_end != 0 || pw_end != 0;
    let get_kernel = if !need_pad {
        T::get_kernel
    } else {
        T::get_kernel_with_padding
    };

    for g in 0..groups {
        pack_kernel(
            buffer.ptr(),
            kernel_ptr + g * out_channels_per_group,
            in_channels_per_group,
            out_channels_per_group,
            ic,
            oc,
            nr as i64,
            [kernel_height, kernel_width],
            [ks0, ks1, ks2],
        );
        (0..outer).into_par_iter().for_each(|idx| {
            let kernel = buffer.ptr();
            let b = idx / out_height;
            let ll = idx % out_height;

            let inp = inp_ptr.clone() + b * isb + g * in_channels_per_group;
            let out = out.clone() + b * osb + ll * osh + g * out_channels_per_group;

            for k in (0..out_width).step_by(kc as usize) {
                let owb = kc.min(out_width - k);
                let mut kernel_idx: i64 = 0;
                for i in (0..in_channels_per_group).step_by(ic as usize) {
                    let icb = ic.min(in_channels_per_group - i);
                    for j in (0..out_channels_per_group).step_by(oc as usize) {
                        let ocb = oc.min(out_channels_per_group - j);

                        let kernel_idx_1 = kernel_idx;
                        for kk in (0..owb).step_by(mr as usize) {
                            let owr = (mr as i64).min(owb - kk);
                            let micro_kernel = get_kernel(nr / <T>::Vec::SIZE, owr as usize);
                            kernel_idx = kernel_idx_1;
                            for jj in (0..ocb).step_by(nr as usize) {
                                let ocr = (nr as i64).min(ocb - jj);
                                micro_kernel(
                                    inp + i,
                                    kernel,
                                    out,
                                    icb,
                                    osw,
                                    &mut kernel_idx,
                                    [kk + k, jj + j, ll],
                                    [kernel_height, kernel_width],
                                    [step_height, step_width],
                                    [ph_start, pw_start],
                                    [img_height, img_width],
                                    [ish, isw],
                                    [owr, ocr],
                                    [dh, dw],
                                    i == 0,
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    handle_post(&mut output, bias, post_scalar, post_vec)?;

    Ok(output)
}
