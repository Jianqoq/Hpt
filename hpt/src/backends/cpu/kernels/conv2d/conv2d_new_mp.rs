use crate::backends::common::conv::cal_conv2d_output_shape;
use crate::tensor_base::_Tensor;
use hpt_allocator::traits::{ Allocator, AllocatorOutputRetrive };
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::NormalOutPromote;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::calculate_kernel_params;
use super::utils::create_packed_kernel;
use super::utils::pack_kernel_mp;

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel, const DEVICE: usize, A>(
    input: &_Tensor<T, Cpu, DEVICE, A>,
    kernels: &_Tensor<T, Cpu, DEVICE, A>,
    bias: Option<&_Tensor<T, Cpu, DEVICE, A>>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2]
)
    -> Result<_Tensor<T, Cpu, DEVICE, A>, TensorError>
    where
    bool: Cast<T>,
        A: Allocator + Send + Sync,
        A::Output: AllocatorOutputRetrive,
        T: Cast<<T as NormalOutPromote>::Intermediate>,
        <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>
{
    ShapeError::check_contiguous(
        "Conv2d requires input tensor to be contiguous. ".to_string(),
        input.layout()
    )?;
    ShapeError::check_contiguous(
        "Conv2d requires kernel tensor to be contiguous. ".to_string(),
        kernels.layout()
    )?;
    if bias.is_some() {
        ShapeError::check_contiguous(
            "Conv2d requires bias tensor to be contiguous. ".to_string(),
            bias.unwrap().layout()
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
        return Err(
            (ShapeError::ConvError {
                message: format!(
                    "kernel in_channel {} not match input in_channel {}",
                    in_channels,
                    img_channels
                ),
                location: core::panic::Location::caller(),
            }).into()
        );
    }
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
    if out_height <= 0 || out_width <= 0 {
        return Err(
            (ShapeError::ConvError {
                message: if out_height <= 0 {
                    "output height <= 0".to_string()
                } else {
                    "output width <= 0".to_string()
                },
                location: core::panic::Location::caller(),
            }).into()
        );
    }
    let output = _Tensor::<T, Cpu, DEVICE, A>::empty([batch, out_height, out_width, out_channels])?;
    let out = output.ptr();

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

    let inp_ptr = input.ptr();
    let kernel_ptr = kernels.ptr();
    let nr = T::get_max_mixed_precision_nr() * T::Vec::SIZE;
    let mr = T::get_max_mixed_precision_mr().min(out_width as usize);
    let param = calculate_kernel_params::<T>(in_channels, out_channels, out_width, mr, nr, [kh as usize, kw as usize]);

    let kc: i64 = param.nc as i64;
    let ic: i64 = param.kc as i64;
    let oc: i64 = param.mc as i64;
    let buffer =
        create_packed_kernel::<<T as NormalOutPromote>::Intermediate, DEVICE, A>(
            kh,
            kw,
            in_channels,
            out_channels,
            oc,
            nr as i64
        )?;
    pack_kernel_mp(
        buffer.ptr(),
        kernel_ptr,
        in_channels,
        out_channels,
        ic,
        oc,
        nr as i64,
        [kh, kw],
        [ks0, ks1, ks2]
    );

    let get_kernel = if ph_start == 0 && pw_start == 0 && ph_end == 0 && pw_end == 0 {
        T::get_mixed_precision_kernel
    } else {
        T::get_mixed_precision_kernel_with_padding
    };

    (0..outer).into_par_iter().for_each(|idx| {
        let kernel = buffer.ptr();
        let b = idx / out_height;
        let ll = idx % out_height;

        let inp = inp_ptr.clone() + b * isb + ll * step_height * ish;
        let out = out.clone() + b * osb + ll * osh;

        for k in (0..out_width).step_by(kc as usize) {
            let owb = kc.min(out_width - k);
            let mut kernel_idx: i64 = 0;
            for i in (0..in_channels).step_by(ic as usize) {
                let icb = ic.min(in_channels - i);
                for j in (0..out_channels).step_by(oc as usize) {
                    let ocb = oc.min(out_channels - j);

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
                                [kh, kw],
                                [step_height, step_width],
                                [ph_start, pw_start],
                                [img_height, img_width],
                                [ish, isw],
                                [owr, ocr],
                                [dh, dw],
                                i == 0,
                                |x| x.cast(),
                                |x| x.cast()
                            );
                        }
                    }
                }
            }
        }
    });

    Ok(output)
}
