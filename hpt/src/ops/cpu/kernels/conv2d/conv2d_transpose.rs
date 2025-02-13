use crate::ops::cpu::kernels::conv_transpose::{
    full_oc_kernel_dispatch, remain_ic_kernel_dispatch, Params, PartialParams,
};
use crate::tensor_base::_Tensor;
use crate::Cpu;
use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::CommonBounds;
use hpt_traits::TensorCreator;
use hpt_traits::TensorInfo;
use hpt_types::into_scalar::Cast;
use hpt_types::vectors::traits::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[track_caller]
pub(crate) fn conv2d_transpose<T: CommonBounds, const DEVICE: usize>(
    input: &_Tensor<T, Cpu, DEVICE>,
    kernels: &_Tensor<T, Cpu, DEVICE>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
) -> Result<_Tensor<T, Cpu, DEVICE>, TensorError>
where
    bool: Cast<T>,
{
    ShapeError::check_contiguous(
        "conv2d_transpose input must be contiguous".to_string(),
        &input.layout(),
    )?;
    let inp_shape = input.shape();
    ShapeError::check_dim(4, inp_shape.len())?;
    let batch = inp_shape[0];
    let inp_height = inp_shape[1];
    let inp_width = inp_shape[2];
    let inp_channels = inp_shape[3];
    let kernel_shape = kernels.shape();
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let in_channel = kernel_shape[2];
    let out_channel = kernel_shape[3];
    if out_channel != inp_channels {
        return Err(ShapeError::ConvError {
            message: format!(
                "kernel in_channel {} not match input in_channel {}",
                in_channel, out_channel
            ),
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let out_height = (inp_height - 1) * step_height - (ph_start + ph_end)
        + dh * (kh - 1)
        + 1
        + output_padding[0];
    let out_width =
        (inp_width - 1) * step_width - (pw_start + pw_end) + dw * (kw - 1) + 1 + output_padding[1];
    let res = _Tensor::<T, Cpu, DEVICE>::zeros([batch, out_height, out_width, in_channel])?;
    let out = res.ptr();
    let inp = input.ptr();

    let osb = input.strides()[0]; // batch
    let osh = input.strides()[1]; // height
    let osw = input.strides()[2]; // width

    let isb = res.strides()[0]; // batch
    let ish = res.strides()[1]; // height
    let isw = res.strides()[2]; // width

    let ks0 = kernels.strides()[0]; // kernel_height
    let ks1 = kernels.strides()[1]; // kernel_width
    let ks2 = kernels.strides()[2]; // in_channels

    let mut iw_block = 1;
    let mut oc_block = 4;
    let ic_block = 4;
    let ih_block = 5;
    let full_ic = full_oc_kernel_dispatch(&mut oc_block, &mut iw_block);
    let remain_ic = remain_ic_kernel_dispatch(&mut iw_block);
    let mut remain = inp_width as usize % iw_block;
    let full_ic_remain_ow = full_oc_kernel_dispatch(&mut oc_block, &mut remain);
    let remain_ic_remain_ow = remain_ic_kernel_dispatch(&mut remain);

    let ro_kernel = kernels.empty_like()?;
    let mut ro_ptr = ro_kernel.ptr();

    for oo in (0..out_channel).step_by(T::Vec::SIZE * oc_block) {
        let o_end = (oo + ((T::Vec::SIZE * oc_block) as i64)).min(out_channel);
        for i in (0..in_channel).step_by(T::Vec::SIZE * ic_block) {
            let i_start = i;
            let i_end = (i + (T::Vec::SIZE * ic_block) as i64).min(in_channel);
            let remain = (i_end - i_start) % (T::Vec::SIZE * ic_block) as i64;
            if remain > 0 {
                for n in 0..kh {
                    for m in 0..kw {
                        for o in oo..o_end {
                            for j in 0..in_channel % (T::Vec::SIZE * ic_block) as i64 {
                                let j = i + j;
                                let kr = kernels.ptr()[n * ks0 + m * ks1 + o + j * ks2];
                                let ptr = ro_ptr.ptr;
                                unsafe { ptr.write(kr) };
                                ro_ptr += 1usize;
                            }
                        }
                    }
                }
            } else {
                for n in 0..kh {
                    for m in 0..kw {
                        for o in oo..o_end {
                            for j in (0..(T::Vec::SIZE * ic_block) as i64).step_by(T::Vec::SIZE) {
                                let j = i + j;
                                for jj in 0..T::Vec::SIZE as i64 {
                                    let kr = kernels.ptr()[n * ks0 + m * ks1 + o + (j + jj) * ks2];
                                    let ptr = ro_ptr.ptr;
                                    unsafe { ptr.write(kr) };
                                    ro_ptr += 1usize;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let ic_block_size = ic_block * T::Vec::SIZE;
    let num_ih = (inp_height + ih_block as i64 - 1) / ih_block as i64; // div ceil, i.e. ceiling of out_height / oh_block
    let outer = batch * num_ih;
    (0..outer).into_iter().for_each(|idx| {
        let inp = inp.clone();
        let mut out = out.clone();
        let mut kernel = ro_kernel.ptr();
        let b = idx / num_ih;
        let ll = idx % num_ih;
        let ll = ll * ih_block as i64;
        let l_end = (ll + ih_block as i64).min(inp_height);
        for oo in (0..out_channel).step_by(T::Vec::SIZE * oc_block) {
            let o_end = (oo + ((T::Vec::SIZE * oc_block) as i64)).min(out_channel);
            let original_k = kernel.clone();
            for k in (0..inp_width).step_by(iw_block) {
                let k_start = k;
                let k_end = (k + iw_block as i64).min(inp_width);
                let k_remain = (k_end - k_start) % (iw_block) as i64;
                if k_remain == 0 {
                    for i in (0..in_channel).step_by(T::Vec::SIZE * ic_block) {
                        let i_start = i;
                        let i_end = (i + (T::Vec::SIZE * ic_block) as i64).min(in_channel);
                        let remain = (i_end - i_start) % (T::Vec::SIZE * ic_block) as i64;
                        if remain > 0 {
                            let original = kernel.clone();
                            for l in ll..l_end {
                                let param = PartialParams {
                                    arg1: [oo, o_end],
                                    arg2: [kh, kw],
                                    arg3: [b, l, k, i],
                                    arg4: [osb, osh, osw],
                                    arg5: [step_height, step_width],
                                    arg6: [isb, ish, isw],
                                    arg7: [ph_start, pw_start],
                                    arg8: [dh, dw],
                                    arg9: [out_height, out_width],
                                    ic_remain: in_channel % (T::Vec::SIZE * ic_block) as i64,
                                };
                                remain_ic(param, &mut out, &mut kernel, &inp);
                                kernel = original.clone();
                            }
                            kernel += kh * kw * (o_end - oo) * remain;
                        } else {
                            let original = kernel.clone();
                            for l in ll..l_end {
                                let param = Params {
                                    arg1: [oo, o_end],
                                    arg2: [kh, kw],
                                    arg3: [b, l, k, i],
                                    arg4: [osb, osh, osw],
                                    arg5: [step_height, step_width],
                                    arg6: [isb, ish, isw],
                                    pads: [ph_start, pw_start],
                                    arg8: [dh, dw],
                                    arg9: [out_height, out_width],
                                };
                                full_ic(param, &mut out, &mut kernel, &inp);
                                kernel = original.clone();
                            }
                            kernel += kh * kw * (o_end - oo) * (ic_block_size as i64);
                        }
                    }
                } else {
                    for i in (0..in_channel).step_by(T::Vec::SIZE * ic_block) {
                        if i + (ic_block * T::Vec::SIZE) as i64 <= in_channel {
                            let original = kernel.clone();
                            for l in ll..l_end {
                                let param = Params {
                                    arg1: [oo, o_end],
                                    arg2: [kh, kw],
                                    arg3: [b, l, k, i],
                                    arg4: [osb, osh, osw],
                                    arg5: [step_height, step_width],
                                    arg6: [isb, ish, isw],
                                    pads: [ph_start, pw_start],
                                    arg8: [dh, dw],
                                    arg9: [out_height, out_width],
                                };
                                full_ic_remain_ow(param, &mut out, &mut kernel, &inp);
                                kernel = original.clone();
                            }
                            kernel += kh * kw * (o_end - oo) * (ic_block_size as i64);
                        } else {
                            let original = kernel.clone();
                            for l in ll..l_end {
                                let param = PartialParams {
                                    arg1: [oo, o_end],
                                    arg2: [kh, kw],
                                    arg3: [b, l, k, i],
                                    arg4: [osb, osh, osw],
                                    arg5: [step_height, step_width],
                                    arg6: [isb, ish, isw],
                                    arg7: [ph_start, pw_start],
                                    arg8: [dh, dw],
                                    arg9: [out_height, out_width],
                                    ic_remain: in_channel % (T::Vec::SIZE * ic_block) as i64,
                                };
                                remain_ic_remain_ow(param, &mut out, &mut kernel, &inp);
                                kernel = original.clone();
                            }
                            kernel += kh
                                * kw
                                * (o_end - oo)
                                * (in_channel % (T::Vec::SIZE * ic_block) as i64);
                        }
                    }
                }
                kernel = original_k.clone();
            }
            kernel += kh * kw * (o_end - oo) * in_channel;
        }
    });
    Ok(res)
}

#[allow(unused)]
#[track_caller]
pub(crate) fn conv2d_backward_kernel<T: CommonBounds, const DEVICE: usize>(
    input: &_Tensor<T, Cpu, DEVICE>,
    grad_output: &_Tensor<T, Cpu, DEVICE>,
    kernel_shape: &[i64],
    stride: [i64; 2],
    padding: [(i64, i64); 2],
) -> Result<_Tensor<T, Cpu, DEVICE>, TensorError> {
    let inp_shape = input.shape();
    let grad_shape = grad_output.shape();
    let batch = inp_shape[0];
    let inp_height = inp_shape[1];
    let inp_width = inp_shape[2];
    let in_channel = inp_shape[3];
    let out_channel = grad_shape[3];

    // kernel梯度形状: [kernel_height, kernel_width, in_channel, out_channel]
    let grad_kernel = _Tensor::<T, Cpu, DEVICE>::zeros(kernel_shape)?;

    let (sh, sw) = (stride[0], stride[1]);
    let ((ph_start, _), (pw_start, _)) = (padding[0], padding[1]);

    let input_ptr = input.ptr();
    let grad_output_ptr = grad_output.ptr();

    let ibs = input.strides()[0];
    let ish = input.strides()[1];
    let isw = input.strides()[2];

    let gbs = grad_output.strides()[0];
    let gsh = grad_output.strides()[1];
    let gsw = grad_output.strides()[2];

    let gkbs = grad_kernel.strides()[0];
    let gksh = grad_kernel.strides()[1];
    let gksw = grad_kernel.strides()[2];

    let outer = batch * grad_shape[1] * grad_shape[2];

    let grad_kernel_ptr = grad_kernel.ptr();
    (0..outer).into_par_iter().for_each(|idx| {
        let mut grad_kernel_ptr = grad_kernel_ptr.clone();
        let b = idx / (grad_shape[1] * grad_shape[2]);
        let h_out = (idx % (grad_shape[1] * grad_shape[2])) / grad_shape[2];
        let w_out = idx % grad_shape[2];
        for kh in 0..kernel_shape[0] {
            for kw in 0..kernel_shape[1] {
                let h_in = h_out * sh + kh - ph_start;
                let w_in = w_out * sw + kw - pw_start;

                if h_in >= 0 && h_in < inp_height && w_in >= 0 && w_in < inp_width {
                    for ic in 0..in_channel {
                        for oc in 0..out_channel {
                            let inp_val = input_ptr[b * ibs + h_in * ish + w_in * isw + ic];

                            let grad_out_val =
                                grad_output_ptr[b * gbs + h_out * gsh + w_out * gsw + oc];

                            let gk_idx = kh * gkbs + kw * gksh + ic * gksw + oc;
                            grad_kernel_ptr[gk_idx] =
                                inp_val._mul_add(grad_out_val, grad_kernel_ptr[gk_idx]);
                        }
                    }
                }
            }
        }
    });
    Ok(grad_kernel)
}

#[allow(unused)]
#[track_caller]
pub(crate) fn conv2d_backward_bias<T: CommonBounds, const DEVICE: usize>(
    grad_output: &_Tensor<T, Cpu, DEVICE>,
) -> Result<_Tensor<T, Cpu, DEVICE>, TensorError> {
    let grad_shape = grad_output.shape();
    let batch = grad_shape[0];
    let out_height = grad_shape[1];
    let out_width = grad_shape[2];
    let out_channel = grad_shape[3];

    // bias梯度形状: [out_channel]
    let grad_bias = _Tensor::<T, Cpu, DEVICE>::zeros([out_channel])?;

    let mut grad_bias_ptr = grad_bias.ptr();
    let gbs = grad_output.strides()[0];
    let gsh = grad_output.strides()[1];
    let gsw = grad_output.strides()[2];
    // 对每个输出通道，累加所有位置的梯度
    for oc in 0..out_channel {
        let mut sum = T::ZERO;
        for b in 0..batch {
            for h in 0..out_height {
                for w in 0..out_width {
                    sum = sum._add(grad_bias_ptr[b * gbs + h * gsh + w * gsw + oc]);
                }
            }
        }
        grad_bias_ptr[oc] = sum;
    }
    Ok(grad_bias)
}
