use hpt_common::error::base::TensorError;
use hpt_matmul::PrePackedRhs;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::ToDType;
use hpt_types::vectors::traits::*;
use rayon::prelude::*;

use crate::Tensor;

use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::create_packed_kernel_raw;
use super::utils::{calculate_kernel_params, create_packed_kernel, handle_post, pack_kernel};

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel + ToDType>(
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
    prepacked_rhs: Option<&PrePackedRhs>,
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    mut output: Tensor,
) -> Result<Tensor, TensorError> {
    let in_channels = img_channels;
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let (out_height, out_width) = (output.shape()[1] as i64, output.shape()[2] as i64);
    let img = input.clone();
    let out = output.ptr::<T>();

    let osb = output.strides()[0] as i64; // batch
    let osh = output.strides()[1] as i64; // height
    let osw = output.strides()[2] as i64; // width

    let isb = img.strides()[0] as i64; // batch
    let ish = img.strides()[1] as i64; // height
    let isw = img.strides()[2] as i64; // width

    let ks0 = kernels.strides()[0] as i64; // kernel_height
    let ks1 = kernels.strides()[1] as i64; // kernel_width
    let ks2 = kernels.strides()[2] as i64; // in_channels
    let ks3 = kernels.strides()[3] as i64; // out_channels

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
        [kh as usize, kw as usize],
    );
    let oc: i64 = param.nc as i64;
    let ic: i64 = param.kc as i64;
    let kc: i64 = param.mc as i64;
    let buffer = create_packed_kernel(
        input.dtype,
        input.device.clone(),
        kh,
        kw,
        in_channels,
        out_channels,
        oc,
        nr as i64,
    )?;
    pack_kernel(
        buffer.ptr(),
        kernel_ptr,
        in_channels,
        out_channels,
        ic,
        oc,
        nr as i64,
        [kh, kw],
        [ks0, ks1, ks2, ks3],
    );
    let need_pad = ph_start != 0 || pw_start != 0 || ph_end != 0 || pw_end != 0;
    let get_kernel = if !need_pad {
        T::get_kernel
    } else {
        T::get_kernel_with_padding
    };

    (0..outer).into_par_iter().for_each(|idx| {
        let kernel = buffer.ptr();
        let b = idx / out_height;
        let ll = idx % out_height;

        let inp = inp_ptr + b * isb;
        let out = out + b * osb + ll * osh;

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
                            );
                        }
                    }
                }
            }
        }
    });

    handle_post(&mut output, bias, post_scalar, post_vec)?;

    Ok(output)
}

pub(crate) fn conv2d_prepack_kernel<T: CommonBounds + Conv2dMicroKernel + ToDType>(
    input: &Tensor,
    kernels: &Tensor,
    img_channels: i64,
    out_channels: i64,
    kh: i64,
    kw: i64,
) -> Result<PrePackedRhs, TensorError> {
    let in_channels = img_channels;

    let ks0 = kernels.strides()[0] as i64; // kernel_height
    let ks1 = kernels.strides()[1] as i64; // kernel_width
    let ks2 = kernels.strides()[2] as i64; // in_channels
    let ks3 = kernels.strides()[3] as i64; // out_channels

    let kernel_ptr = kernels.ptr::<T>();
    let nr = T::get_max_nr() * T::Vec::SIZE;
    let mr = T::get_max_mr();
    let param = calculate_kernel_params::<T>(
        in_channels,
        out_channels,
        0,
        mr,
        nr,
        [kh as usize, kw as usize],
    );
    let oc: i64 = param.nc as i64;
    let ic: i64 = param.kc as i64;
    let (buffer, layout) = create_packed_kernel_raw(
        input.dtype,
        kh,
        kw,
        in_channels,
        out_channels,
        oc,
        nr as i64,
    );
    pack_kernel(
        buffer.cast::<T>(),
        kernel_ptr,
        in_channels,
        out_channels,
        ic,
        oc,
        nr as i64,
        [kh, kw],
        [ks0, ks1, ks2, ks3],
    );

    Ok(PrePackedRhs {
        buffer: todo!(),
        kc: ic as usize,
        nr: nr as usize,
        nc: oc as usize,
    })
}
