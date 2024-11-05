use crate::ops::cpu::cache_utils::cache::Cache;
use crate::ops::cpu::kernels::transpose_conv2d::bias_remain_oc_kernel_dispatch;
use crate::ops::cpu::kernels::transpose_conv2d::conv2d_full_oc_bias_kernel_dispatch;
use crate::ops::cpu::kernels::transpose_conv2d::tconv2d_full_ic_dispatch;
use crate::ops::cpu::kernels::transpose_conv2d::remain_oc_kernel_dispatch;
use crate::ops::cpu::kernels::transpose_conv2d::Params;
use crate::ops::cpu::kernels::transpose_conv2d::PartialParams;
use crate::tensor_base::_Tensor;
use crate::REGNUM;
use crate::SIMD_WIDTH;
use rayon::prelude::*;
use tensor_common::err_handler::ErrHandler;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;
use tensor_types::vectors::traits::*;

impl<T> _Tensor<T>
    where
        T: CommonBounds + IntoScalar<T> + NormalOut<Output = T>,
        T::Vec: VecTrait<T> + Copy + Init<T> + Send + Sync + VecCommon + NormalOut<Output = T::Vec>,
        bool: IntoScalar<T>
{
    /// Performs a 2D convolution operation on the input tensor.
    ///
    /// This method applies a 2D convolution operation on the tensor using the specified kernel,
    /// strides (steps), padding, and dilation factors.
    ///
    /// # Arguments
    ///
    /// * `kernels` - A reference to the tensor representing the convolution kernels (filters).
    ///   The size of the kernel tensor determines the spatial dimensions of the convolution operation.
    /// * `steps` - A 2-element array specifying the stride (step size) of the convolution along the height and width dimensions.
    /// * `padding` - A 2-element array of tuples representing the padding for the height and width dimensions.
    ///   Each tuple specifies the amount of padding added before and after the data along the respective axis.
    /// * `dilation` - A 2-element array specifying the dilation factor for the convolution along the height and width dimensions.
    ///   Dilation allows the kernel to be applied to inputs with gaps, increasing the receptive field of the kernel.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the 2D convolution operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    #[inline(never)]
    pub fn transpose_conv2d(
        &self,
        kernels: &_Tensor<T>,
        bias: Option<&_Tensor<T>>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T) -> T>
    ) -> anyhow::Result<_Tensor<T>> {
        let img_shape = self.shape();
        if img_shape.len() != 4 {
            return Err(
                ErrHandler::Conv2dImgShapeInCorrect(
                    img_shape.len(),
                    core::panic::Location::caller()
                ).into()
            );
        }
        let batch = img_shape[0];
        let img_height = img_shape[1];
        let img_width = img_shape[2];
        let img_channels = img_shape[3];
        let kernel_shape = kernels.shape();
        let kh = kernel_shape[0];
        let kw = kernel_shape[1];
        let out_channels = kernel_shape[2];
        let in_channels = kernel_shape[3];
        if in_channels != img_channels {
            panic!(
                "The number of input channels in the image must be equal to the number of input channels in the kernel."
            );
        }
        let (step_width, step_height) = (steps[0], steps[1]);
        let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let out_height = (img_height - 1) * step_height + dh * (kh - 1) + 1 - ph_start - ph_end;
        let out_width = (img_width - 1) * step_width + dw * (kw - 1) + 1 - pw_start - pw_end;
        let img = self.clone();
        if out_height <= 0 || out_width <= 0 {
            return if out_height <= 0 {
                Err(InvalidInputShape(out_height, core::panic::Location::caller()).into())
            } else {
                Err(InvalidInputShape(out_width, core::panic::Location::caller()).into())
            };
        }
        let activation = activation.unwrap_or(|x| x);
        let output = _Tensor::<T>::zeros([batch, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp = img.ptr();

        let isb = img.strides()[0]; // batch
        let ish = img.strides()[1]; // height
        let isw = img.strides()[2]; // width

        let osb = output.strides()[0]; // batch
        let osh = output.strides()[1]; // height
        let osw = output.strides()[2]; // width

        let ks0 = kernels.strides()[0]; // kernel_height
        let ks1 = kernels.strides()[1]; // kernel_width
        let ks2 = kernels.strides()[2]; // in_channels

        let ih_block = (3).min(img_height).max(1);

        let cache = Cache::<T>::new();

        let mut ic_nvec = cache.l1_line_size / T::Vec::SIZE;
        let mut iw_block = predict_iw_block(ic_nvec);

        let params = kernel_params::<T>(
            in_channels as usize,
            out_channels as usize,
            iw_block,
            ic_nvec,
            ih_block as usize,
            [kh as usize, kw as usize],
            cache
        );
        let (oc_nvec, ib) = params;

        // retrieve micro kernels start

        let full_oc_kernel = tconv2d_full_ic_dispatch(
            [kh, kw],
            &mut ic_nvec,
            &mut iw_block
        ).expect(&format!("unable to find iconv2d_microkernel_{}x{}", iw_block, ic_nvec));
        let full_ic_kernel_fn = full_oc_kernel.kernel.clone();
        let full_ic_kernel_iw_remain = tconv2d_full_ic_dispatch::<T>(
            [kh, kw],
            &mut ic_nvec,
            &mut ((img_width as usize) % iw_block)
        );
        if full_ic_kernel_iw_remain.is_none() {
            assert_eq!((img_width as usize) % iw_block, 0);
        }
        let partial_ic_kernel = remain_oc_kernel_dispatch::<T>([kh, kw], &mut iw_block);
        if let Some(partial_oc_kernel) = partial_ic_kernel {
            assert_eq!(iw_block, partial_oc_kernel.iw_block);
        }
        let partial_ic_kernel_iw_remain = remain_oc_kernel_dispatch::<T>(
            [kh, kw],
            &mut ((img_width as usize) % iw_block)
        );
        if partial_ic_kernel_iw_remain.is_none() {
            assert_eq!((img_width as usize) % iw_block, 0);
        }
        let full_ic_kernel_fn_1_ic = tconv2d_full_ic_dispatch::<T>(
            [kh, kw],
            &mut 1,
            &mut iw_block
        );
        let full_ic_kernel_fn_1_ic_iw_remain = tconv2d_full_ic_dispatch::<T>(
            [kh, kw],
            &mut 1,
            &mut ((img_width as usize) % iw_block)
        );
        let has_bias = bias.is_some();
        let bias_full_ic_kernel = if has_bias {
            Some(
                conv2d_full_oc_bias_kernel_dispatch::<T>(
                    [kh, kw],
                    &mut ic_nvec,
                    &mut iw_block
                ).unwrap()
            )
        } else {
            None
        };
        let bias_one_ic_kernel = if has_bias {
            Some(conv2d_full_oc_bias_kernel_dispatch::<T>([kh, kw], &mut 1, &mut iw_block).unwrap())
        } else {
            None
        };
        let bias_remain_ic_kernel = if has_bias {
            Some(bias_remain_oc_kernel_dispatch::<T>([kh, kw], &mut iw_block).unwrap())
        } else {
            None
        };
        let bias_full_ic_iw_remain = if has_bias {
            Some(
                conv2d_full_oc_bias_kernel_dispatch::<T>(
                    [kh, kw],
                    &mut ic_nvec,
                    &mut ((img_width as usize) % iw_block)
                ).unwrap()
            )
        } else {
            None
        };
        let bias_one_ic_iw_remain = if has_bias {
            Some(
                conv2d_full_oc_bias_kernel_dispatch::<T>(
                    [kh, kw],
                    &mut 1,
                    &mut ((img_width as usize) % iw_block)
                ).unwrap()
            )
        } else {
            None
        };
        let bias_partial_ic_iw_remain = if has_bias {
            Some(
                bias_remain_oc_kernel_dispatch::<T>(
                    [kh, kw],
                    &mut ((img_width as usize) % iw_block)
                ).unwrap()
            )
        } else {
            None
        };

        // retrieve micro kernels end

        let num_ih = (img_height + ih_block - 1) / ih_block; // div ceil, i.e. ceiling of out_height / oh_block
        let outer = batch * num_ih;
        let out_width_full_end = img_width - (img_width % (iw_block as i64)); // the end of the out width that is a multiple of ow_block

        // create new memory space to store the reordered kernel filter
        let ro_kernel = kernels.empty_like()?;
        let ro_ptr = ro_kernel.ptr();

        // reorder the kernel filter, so that when we do convolution, we can simply increment the pointer and get the data, this can significantly reduce cache miss rate and improve performance
        reorder_kernel(
            &kernels.ptr(),
            ro_ptr.clone(),
            ib,
            [out_channels as usize, oc_nvec],
            [in_channels as usize, ic_nvec],
            [ks0 as usize, ks1 as usize, ks2 as usize],
            [kh as usize, kw as usize]
        );

        let oc_block_size = oc_nvec * T::Vec::SIZE; // in channel block size
        let ic_block_size = ic_nvec * T::Vec::SIZE; // out channel block size, but this is only caculated based on cache line size, we have another block size `jb`
        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let mut kernel = ro_ptr.clone();
            let b = idx / num_ih;
            let ll = idx % num_ih;
            let ll = ll * ih_block;
            let l_end = (ll + ih_block).min(img_height);
            let params = Params {
                arg1: [0, 0],
                arg2: [kh, kw],
                arg3: [b, 0, 0, 0],
                arg4: [isb, ish, isw],
                arg5: [step_height, step_width],
                arg6: [osb, osh, osw],
                pads: [ph_start, pw_start],
                arg8: [dh, dw],
                arg9: [out_height, out_width],
            };
            match (partial_ic_kernel, full_ic_kernel_iw_remain, partial_ic_kernel_iw_remain) {
                (None, None, None) => {
                    for jj in (0..out_channels).step_by(oc_block_size) {
                        let jj_end = (jj + (oc_block_size as i64)).min(out_channels);
                        let end = jj_end == out_channels;
                        let params = Params {
                            arg1: [jj, jj_end],
                            ..params
                        };
                        if end {
                            if has_bias {
                                let bias = bias.unwrap().ptr();
                                let bias_full_oc_kernel = bias_full_ic_kernel.unwrap();
                                conv_perfect(
                                    in_channels,
                                    ic_block_size,
                                    iw_block,
                                    out_width_full_end,
                                    [kh, kw],
                                    [ll, l_end],
                                    jj_end - jj,
                                    ib,
                                    &mut kernel,
                                    &mut out,
                                    |l, k, i, kernel, out| {
                                        bias_full_oc_kernel(
                                            Params {
                                                arg3: [b, l, k, i],
                                                ..params
                                            },
                                            out,
                                            kernel,
                                            &inp,
                                            &bias,
                                            activation
                                        );
                                    }
                                );
                            } else {
                                conv_perfect(
                                    in_channels,
                                    ic_block_size,
                                    iw_block,
                                    out_width_full_end,
                                    [kh, kw],
                                    [ll, l_end],
                                    jj_end - jj,
                                    ib,
                                    &mut kernel,
                                    &mut out,
                                    |l, k, i, kernel, out| {
                                        full_ic_kernel_fn(
                                            Params {
                                                arg3: [b, l, k, i],
                                                ..params
                                            },
                                            out,
                                            kernel,
                                            &inp,
                                            activation
                                        );
                                    }
                                );
                            }
                        } else {
                            conv_perfect(
                                in_channels,
                                ic_block_size,
                                iw_block,
                                out_width_full_end,
                                [kh, kw],
                                [ll, l_end],
                                jj_end - jj,
                                ib,
                                &mut kernel,
                                &mut out,
                                |l, k, i, kernel, out| {
                                    full_ic_kernel_fn(
                                        Params {
                                            arg3: [b, l, k, i],
                                            ..params
                                        },
                                        out,
                                        kernel,
                                        &inp,
                                        |x| x
                                    );
                                }
                            );
                        }
                    }
                }
                _ => {
                    for jj in (0..out_channels).step_by(oc_block_size) {
                        let jj_end = (jj + (oc_block_size as i64)).min(out_channels);
                        if jj_end == out_channels && has_bias {
                            let bias = bias.unwrap().ptr();
                            let bias_full_ic_kernel = bias_full_ic_kernel.unwrap();
                            // out channel has two levels of blocking:
                            // 1. it first blocks by oc_block_size * jb
                            // 2. it then blocks by oc_block_size (cache line size)
                            for ii in (0..in_channels).step_by(ic_block_size * (ib as usize)) {
                                // make sure jj_end is in the range of out_channels
                                let ii_end = (ii + (ic_block_size as i64) * (ib as i64)).min(
                                    in_channels
                                );

                                // calculate the remain part that are less than T::Vec::SIZE * oc_nvec
                                let remain = (ii_end - ii) % (ic_block_size as i64);
                                if remain > 0 {
                                    handle_bias_remain(
                                        [ii, ii_end],
                                        [in_channels, ic_block_size as i64],
                                        [img_width, iw_block as i64],
                                        [ll, l_end],
                                        [jj, jj_end],
                                        [kh, kw],
                                        [isb, ish, isw],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        [out_height, out_width],
                                        b,
                                        remain,
                                        &mut out,
                                        &mut kernel,
                                        &inp,
                                        &bias,
                                        activation,
                                        bias_full_ic_kernel,
                                        bias_remain_ic_kernel.unwrap(),
                                        bias_one_ic_kernel.unwrap(),
                                        bias_full_ic_iw_remain,
                                        bias_partial_ic_iw_remain,
                                        bias_one_ic_iw_remain
                                    );
                                } else {
                                    handle_bias_normal(
                                        [ii, ii_end],
                                        [img_width, iw_block as i64],
                                        [ll, l_end],
                                        [jj, jj_end],
                                        [kh, kw],
                                        [isb, ish, isw],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        [out_height, out_width],
                                        ic_block_size as i64,
                                        b,
                                        &mut out,
                                        &mut kernel,
                                        &inp,
                                        &bias,
                                        activation,
                                        bias_full_ic_kernel,
                                        bias_full_ic_iw_remain
                                    );
                                }
                            }
                        } else {
                            // out channel has two levels of blocking:
                            // 1. it first blocks by oc_block_size * jb
                            // 2. it then blocks by oc_block_size (cache line size)
                            for ii in (0..in_channels).step_by(ic_block_size * (ib as usize)) {
                                // make sure jj_end is in the range of out_channels
                                let ii_end = (ii + (ic_block_size as i64) * (ib as i64)).min(
                                    in_channels
                                );
                                // calculate the remain part that are less than T::Vec::SIZE * oc_nvec
                                let remain = (ii_end - ii) % (ic_block_size as i64);
                                if remain > 0 {
                                    handle_normal_remain(
                                        [ii, ii_end],
                                        [in_channels, ic_block_size as i64],
                                        [img_width, iw_block as i64],
                                        [ll, l_end],
                                        [jj, jj_end],
                                        [kh, kw],
                                        [isb, ish, isw],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        [out_height, out_width],
                                        b,
                                        remain,
                                        &mut out,
                                        &mut kernel,
                                        &inp,
                                        |x| x,
                                        full_ic_kernel_fn,
                                        partial_ic_kernel.unwrap().kernel,
                                        full_ic_kernel_fn_1_ic.unwrap().kernel,
                                        full_ic_kernel_iw_remain.map(|x| x.kernel),
                                        partial_ic_kernel_iw_remain.map(|x| x.kernel),
                                        full_ic_kernel_fn_1_ic_iw_remain.map(|x| x.kernel)
                                    );
                                } else {
                                    handle_normal(
                                        [ii, ii_end],
                                        [img_width, iw_block as i64],
                                        [ll, l_end],
                                        [ii, ii_end],
                                        [kh, kw],
                                        [isb, ish, isw],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        [out_height, out_width],
                                        b,
                                        ic_block_size,
                                        &mut out,
                                        &mut kernel,
                                        &inp,
                                        |x| x,
                                        full_ic_kernel_fn,
                                        full_ic_kernel_iw_remain.map(|x| x.kernel)
                                    );
                                }
                            }
                        }
                    }
                }
            };
        });
        Ok(output)
    }
}

#[allow(unused)]
fn out_used<T: CommonBounds>(
    lb: usize,
    jb: usize,
    oc_nvec: usize,
    owb: usize,
    line_size: usize
) -> usize {
    let nv = line_size / (SIMD_WIDTH / 8 / core::mem::size_of::<T>());
    lb * jb * oc_nvec.div_ceil(nv) * owb * line_size
}
#[allow(unused)]
fn inp_used<T: CommonBounds>(
    lb: usize,
    owb: usize,
    ic_nvec: usize,
    kh: usize,
    kw: usize,
    step_height: usize,
    step_width: usize,
    line_size: usize
) -> usize {
    let nv = line_size / (SIMD_WIDTH / 8 / core::mem::size_of::<T>());
    let in_range_num_w = (0..owb).take_while(|&idx| idx * step_width < kw).count();
    let in_range_num_h = (0..lb).take_while(|&idx| idx * step_height < kh).count();
    owb *
        ic_nvec.div_ceil(nv) *
        (kh + lb - in_range_num_h) *
        (kw + owb - in_range_num_w) *
        line_size
}
#[allow(unused)]
fn kernel_used<T: CommonBounds>(
    oc_nvec: usize,
    ic_nvec: usize,
    jb: usize,
    kh: usize,
    kw: usize
) -> usize {
    oc_nvec * ic_nvec * T::Vec::SIZE * kh * kw * jb
}

fn reorder_kernel<T: CommonBounds>(
    kernel: &Pointer<T>,
    reordered: Pointer<T>,
    jb: usize,
    [out_channel, oc_nvec]: [usize; 2],
    [in_channel, ic_nvec]: [usize; 2],
    [ks0, ks1, ks2]: [usize; 3],
    [kh, kw]: [usize; 2]
) {
    (0..out_channel)
        .into_par_iter()
        .step_by(T::Vec::SIZE * oc_nvec)
        .for_each(|oo| {
            let o_end = (oo + T::Vec::SIZE * oc_nvec).min(out_channel);
            let mut reordered = reordered.clone() + oo * in_channel * kh * kw;
            for ii in (0..in_channel).step_by(T::Vec::SIZE * ic_nvec * jb) {
                let ii_start = ii;
                let ii_end = (ii + T::Vec::SIZE * ic_nvec * jb).min(in_channel);
                let remain = (ii_end - ii_start) % (T::Vec::SIZE * ic_nvec);
                let ic_remain = remain % T::Vec::SIZE;
                for i in (ii_start..ii_end - remain).step_by(T::Vec::SIZE * ic_nvec) {
                    for n in 0..kh {
                        for m in 0..kw {
                            for o in oo..o_end {
                                for v in 0..ic_nvec {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(T::Vec::from_ptr(
                                            &kernel[o * ks2
                                                + n * ks0
                                                + m * ks1
                                                + i
                                                + v * T::Vec::SIZE],
                                        )); // prettier-ignore
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                }
                if remain > 0 {
                    for i in (in_channel - remain..in_channel - ic_remain).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for o in oo..o_end {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(
                                            T::Vec::from_ptr(
                                                &kernel[n * ks0 + m * ks1 + o * ks2 + i]
                                            )
                                        );
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                    for i in (in_channel - ic_remain..in_channel).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for o in oo..o_end {
                                    let ptr: *mut T = reordered.ptr;
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            &kernel[n * ks0 + m * ks1 + o * ks2 + i] as *const T,
                                            ptr,
                                            ic_remain
                                        );
                                    }
                                    reordered += ic_remain;
                                }
                            }
                        }
                    }
                }
            }
        });
}

fn predict_iw_block(ic_block: usize) -> usize {
    REGNUM / (ic_block + 1)
}

/// calculate sub-optimal in channel block size and out channel block size,
/// to maximize the cache utilization and balance the memory access
fn kernel_params<T: CommonBounds>(
    out_channels: usize,
    in_channels: usize,
    ow_block: usize,
    oc_nvec: usize,
    oh_block: usize,
    [kh, kw]: [usize; 2],
    cache: Cache<T>
) -> (usize, usize) {
    let l1 = cache.l1;
    let l2 = cache.l2;

    let ic_range = 1..(in_channels as usize) / T::Vec::SIZE;
    let jb_range = 1..(out_channels as usize) / (T::Vec::SIZE * oc_nvec);

    let best_params = ic_range
        .into_par_iter()
        .flat_map(|ic|
            jb_range
                .clone()
                .into_par_iter()
                .map(move |jb| (ic, jb))
        )
        .filter_map(|(ic, jb)| {
            let gemm_kernel_used = oc_nvec * T::Vec::SIZE * ic * T::Vec::SIZE;
            let gemm_inp_used = ow_block * ic * T::Vec::SIZE;
            let gemm_out_used = ow_block * oc_nvec * T::Vec::SIZE;
            let gemm_used = gemm_kernel_used + gemm_inp_used + gemm_out_used;
            let k_kernel_used = gemm_kernel_used * kh * kw;
            let k_inp_used = kh * kw * gemm_inp_used;
            let jb_k_kernel_used = jb * k_kernel_used;
            let jb_out_used = jb * gemm_out_used;
            let jb_inp_used = oh_block * k_inp_used;
            let total_used = jb_k_kernel_used + jb_out_used + jb_inp_used;

            if gemm_used <= l1 && total_used <= l2 {
                let balance = ((ic as f64) / (jb as f64)).max((jb as f64) / (ic as f64));
                let cache_utilization = (total_used as f64) / (l2 as f64);
                Some((ic, jb, balance, cache_utilization))
            } else {
                None
            }
        })
        .reduce(
            || (1, 1, f64::MAX, 0.0),
            |(best_ic, best_jb, best_balance, best_util), (ic, jb, balance, util)| {
                const BALANCE_WEIGHT: f64 = 0.6;
                const UTIL_WEIGHT: f64 = 0.4;

                let current_score = BALANCE_WEIGHT * (1.0 / balance) + UTIL_WEIGHT * util;
                let best_score = BALANCE_WEIGHT * (1.0 / best_balance) + UTIL_WEIGHT * best_util;

                if current_score > best_score {
                    (ic, jb, balance, util)
                } else {
                    (best_ic, best_jb, best_balance, best_util)
                }
            }
        );

    (best_params.0, best_params.1)
}

fn handle_remain<T: CommonBounds, F, F2, F3>(
    [ii_start, ii_end]: [i64; 2],
    [in_channels, ic_block_size]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    full_ic: F,
    one_ic: F2,
    partial_ic: F3
)
    where
        F: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
        F2: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
        F3: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>)
{
    for j in (ii_start..ii_end - remain).step_by(ic_block_size as usize) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_ic(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (j_end - jj) * (ic_block_size as i64);
    }
    let oc_remain = remain % (T::Vec::SIZE as i64);
    // loop over the remain part that are multiple of T::Vec::SIZE
    for j in (in_channels - remain..in_channels - oc_remain).step_by(T::Vec::SIZE) {
        let original = kernel.clone();
        for l in ll..l_end {
            one_ic(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (T::Vec::SIZE as i64) * (j_end - jj);
    }
    // loop over the remain part that are less than T::Vec::SIZE
    for j in (in_channels - oc_remain..in_channels).step_by(T::Vec::SIZE) {
        let original = kernel.clone();
        for l in ll..l_end {
            partial_ic(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * oc_remain * (j_end - jj);
    }
}

fn _handle_normal<T: CommonBounds, F>(
    [ii_start, ii_end]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    ic_block_size: usize,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    full_ic: F
)
    where F: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>)
{
    for j in (ii_start..ii_end).step_by(ic_block_size) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_ic(j, l, out, kernel);
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (j_end - jj) * (ic_block_size as i64);
    }
}

fn handle_normal<T: CommonBounds, F>(
    [ii_start, ii_end]: [i64; 2],
    [in_width, iw_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    batch: i64,
    ic_block_size: usize,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T) -> T,
    full_ic_kernel_fn: F,
    full_ic_kernel_iw_remain: Option<F>
)
    where F: Fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T)
{
    let params = Params {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    };
    let in_width_full_end = in_width - (in_width % (iw_block as i64));
    let kernel_k = kernel.clone();
    // handle the out width full part
    for k in (0..in_width_full_end).step_by(iw_block as usize) {
        _handle_normal(
            [ii_start, ii_end],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            ic_block_size,
            out,
            kernel,
            |i, l, out, kernel| {
                full_ic_kernel_fn(
                    Params {
                        arg3: [batch, l, k, i],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation
                )
            }
        );
        *kernel = kernel_k.clone();
    }
    // handle the out width remain part
    if let Some(full_ic_kernel_iw_remain) = &full_ic_kernel_iw_remain {
        for k in (in_width_full_end..in_width).step_by(iw_block as usize) {
            _handle_normal(
                [ii_start, ii_end],
                [ll, l_end],
                [jj, j_end],
                [kernel_height, kernel_width],
                ic_block_size,
                out,
                kernel,
                |i, l, out, kernel| {
                    full_ic_kernel_iw_remain(
                        Params {
                            arg3: [batch, l, k, i],
                            ..params
                        },
                        out,
                        kernel,
                        &inp,
                        activation
                    )
                }
            );
            *kernel = kernel_k.clone();
        }
    }
    *kernel += kernel_height * kernel_width * (ii_end - ii_start) * (j_end - jj);
}

fn conv_perfect<T: CommonBounds, F>(
    in_channels: i64,
    ic_block_size: usize,
    iw_block: usize,
    in_width_full_end: i64,
    [kh, kw]: [i64; 2],
    [ll, l_end]: [i64; 2],
    j_range: i64,
    ib: usize,
    kernel: &mut Pointer<T>,
    out: &mut Pointer<T>,
    mut kernel_func: F
)
    where F: FnMut(i64, i64, i64, &mut Pointer<T>, &mut Pointer<T>)
{
    // out channel has two levels of blocking:
    // 1. it first blocks by oc_block_size * jb
    // 2. it then blocks by oc_block_size (cache line size)
    for ii in (0..in_channels).step_by(ic_block_size * (ib as usize)) {
        // make sure jj_end is in the range of out_channels
        let ii_end = (ii + (ic_block_size as i64) * (ib as i64)).min(in_channels);

        // the kernel filter has nothing to do with out height, hence, when iterate over (0..out_width_full_end), kernel pointer should reset in each iteration
        let kernel_k = kernel.clone();

        for k in (0..in_width_full_end).step_by(iw_block) {
            for i in (ii..ii_end).step_by(ic_block_size) {
                // the kernel filter has nothing to do with out height, hence, when iterate over (l..l_end), kernel pointer should reset in each iteration
                let original = kernel.clone();
                for l in ll..l_end {
                    // execute micro kernel, see `iconv2d_full_oc_kernel_dispatch` for more details
                    kernel_func(l, k, i, kernel, out);
                    *kernel = original.clone();
                }
                // update the kernel pointer
                *kernel += kh * kw * j_range * (ic_block_size as i64);
            }
            *kernel = kernel_k.clone();
        }

        *kernel += kh * kw * (ii_end - ii) * j_range;
    }
}

fn with_bias_remain<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [in_channels, ic_block_size]: [i64; 2],
    [start, end, iw_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T) -> T,
    bias_full_ic_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    one_ic: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    partial_ic: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    )
) {
    let params = Params {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    };
    let partial_params = PartialParams {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
        ic_remain: remain % (T::Vec::SIZE as i64),
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(iw_block as usize) {
        handle_remain(
            [ii_start, ii_end],
            [in_channels, ic_block_size as i64],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            remain,
            out,
            kernel,
            |j, l, out, kernel| {
                bias_full_ic_kernel(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation
                )
            },
            |j, l, out, kernel| {
                one_ic(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation
                )
            },
            |j, l, out, kernel| {
                partial_ic(
                    PartialParams {
                        arg3: [batch, l, k, j],
                        ..partial_params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation
                );
            }
        );
        *kernel = kernel_k.clone();
    }
}

fn with_bias_normal<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [start, end, iw_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    ic_block_size: i64,
    batch: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T) -> T,
    bias_full_ic_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    )
) {
    let params = Params {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(iw_block as usize) {
        _handle_normal(
            [ii_start, ii_end],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            ic_block_size as usize,
            out,
            kernel,
            |j, l, out, kernel| {
                bias_full_ic_kernel(
                    Params {
                        arg3: [batch, l, k, j],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    &bias,
                    activation
                )
            }
        );
        *kernel = kernel_k.clone();
    }
}

fn handle_bias_remain<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [in_channels, ic_block_size]: [i64; 2],
    [in_width, iw_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T) -> T,
    bias_full_ic: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    partial_ic: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    bias_one_ic: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    bias_full_ic_iw_remain: Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, &Pointer<T>, fn(T) -> T)
    >,
    bias_partial_ic_iw_remain: Option<
        fn(
            PartialParams,
            &mut Pointer<T>,
            &mut Pointer<T>,
            &Pointer<T>,
            &Pointer<T>,
            fn(T) -> T
        )
    >,
    bias_one_ic_iw_remain: Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, &Pointer<T>, fn(T) -> T)
    >
) {
    let in_width_full_end = in_width - (in_width % (iw_block as i64));
    with_bias_remain(
        [ii_start, ii_end],
        [in_channels, ic_block_size as i64],
        [0, in_width_full_end, iw_block as i64],
        [ll, l_end],
        [jj, j_end],
        [kernel_height, kernel_width],
        [isb, ish, isw],
        [osb, osh, osw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [out_height, out_width],
        batch,
        remain,
        out,
        kernel,
        &inp,
        &bias,
        activation,
        bias_full_ic,
        bias_one_ic,
        partial_ic
    );
    // handle the out width remain part
    if let Some(full_ic_kernel_iw_remain) = &bias_full_ic_iw_remain {
        let one_ic_iw_remain = bias_one_ic_iw_remain.expect(
            &format!("unable to find iconv2d_microkernel_{}x{}", iw_block, 1)
        );
        let partial_ic_iw_remain = bias_partial_ic_iw_remain.expect(
            &format!("unable to find oconv2d_microkernel_{}", iw_block)
        );
        with_bias_remain(
            [ii_start, ii_end],
            [in_channels, ic_block_size as i64],
            [in_width_full_end, in_width, iw_block as i64],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            [isb, ish, isw],
            [osb, osh, osw],
            [step_height, step_width],
            [ph_start, pw_start],
            [dh, dw],
            [out_height, out_width],
            batch,
            remain,
            out,
            kernel,
            &inp,
            &bias,
            activation,
            *full_ic_kernel_iw_remain,
            one_ic_iw_remain,
            partial_ic_iw_remain
        );
    }

    *kernel += kernel_height * kernel_width * (ii_end - ii_start) * (j_end - jj);
}

fn handle_bias_normal<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [in_width, iw_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    oc_block_size: i64,
    batch: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    bias: &Pointer<T>,
    activation: fn(T) -> T,
    bias_full_ic: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    bias_full_ic_iw_remain: Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, &Pointer<T>, fn(T) -> T)
    >
) {
    let out_width_full_end = in_width - (in_width % (iw_block as i64));
    with_bias_normal(
        [ii_start, ii_end],
        [0, out_width_full_end, iw_block as i64],
        [ll, l_end],
        [jj, j_end],
        [kernel_height, kernel_width],
        [isb, ish, isw],
        [osb, osh, osw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [out_height, out_width],
        oc_block_size,
        batch,
        out,
        kernel,
        &inp,
        &bias,
        activation,
        bias_full_ic
    );
    // handle the out width remain part
    if let Some(full_ic_kernel_iw_remain) = &bias_full_ic_iw_remain {
        with_bias_normal(
            [ii_start, ii_end],
            [out_width_full_end, in_width, iw_block as i64],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            [isb, ish, isw],
            [osb, osh, osw],
            [step_height, step_width],
            [ph_start, pw_start],
            [dh, dw],
            [out_height, out_width],
            oc_block_size,
            batch,
            out,
            kernel,
            &inp,
            &bias,
            activation,
            *full_ic_kernel_iw_remain
        );
    }

    *kernel += kernel_height * kernel_width * (ii_end - ii_start) * (j_end - jj);
}

fn with_normal_remain<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [in_channels, ic_block_size]: [i64; 2],
    [start, end, iw_block]: [i64; 3],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T) -> T,
    bias_full_ic_kernel: fn(
        Params,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    one_ic: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T),
    partial_ic: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    )
) {
    let params = Params {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        pads: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
    };
    let partial_params = PartialParams {
        arg1: [jj, j_end],
        arg2: [kernel_height, kernel_width],
        arg3: [batch, 0, 0, 0],
        arg4: [isb, ish, isw],
        arg5: [step_height, step_width],
        arg6: [osb, osh, osw],
        arg7: [ph_start, pw_start],
        arg8: [dh, dw],
        arg9: [out_height, out_width],
        ic_remain: remain % (T::Vec::SIZE as i64),
    };
    let kernel_k = kernel.clone();
    for k in (start..end).step_by(iw_block as usize) {
        handle_remain(
            [ii_start, ii_end],
            [in_channels, ic_block_size as i64],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            remain,
            out,
            kernel,
            |i, l, out, kernel| {
                bias_full_ic_kernel(
                    Params {
                        arg3: [batch, l, k, i],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation
                )
            },
            |i, l, out, kernel| {
                one_ic(
                    Params {
                        arg3: [batch, l, k, i],
                        ..params
                    },
                    out,
                    kernel,
                    &inp,
                    activation
                )
            },
            |i, l, out, kernel| {
                partial_ic(
                    PartialParams {
                        arg3: [batch, l, k, i],
                        ..partial_params
                    },
                    out,
                    kernel,
                    &inp,
                    activation
                );
            }
        );
        *kernel = kernel_k.clone();
    }
}

fn handle_normal_remain<T: CommonBounds>(
    [ii_start, ii_end]: [i64; 2],
    [in_channels, ic_block_size]: [i64; 2],
    [in_width, iw_block]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [jj, j_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    [out_height, out_width]: [i64; 2],
    batch: i64,
    remain: i64,
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    inp: &Pointer<T>,
    activation: fn(T) -> T,
    full_ic: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T),
    partial_ic: fn(
        PartialParams,
        &mut Pointer<T>,
        &mut Pointer<T>,
        &Pointer<T>,
        fn(T) -> T
    ),
    one_ic: fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T),
    full_ic_iw_remain: Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T)
    >,
    partial_ic_iw_remain: Option<
        fn(PartialParams, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T)
    >,
    one_ic_iw_remain: Option<
        fn(Params, &mut Pointer<T>, &mut Pointer<T>, &Pointer<T>, fn(T) -> T)
    >
) {
    let in_width_full_end = in_width - (in_width % (iw_block as i64));
    with_normal_remain(
        [ii_start, ii_end],
        [in_channels, ic_block_size as i64],
        [0, in_width_full_end, iw_block as i64],
        [ll, l_end],
        [jj, j_end],
        [kernel_height, kernel_width],
        [isb, ish, isw],
        [osb, osh, osw],
        [step_height, step_width],
        [ph_start, pw_start],
        [dh, dw],
        [out_height, out_width],
        batch,
        remain,
        out,
        kernel,
        &inp,
        activation,
        full_ic,
        one_ic,
        partial_ic
    );
    // handle the out width remain part
    if let Some(full_ic_kernel_iw_remain) = &full_ic_iw_remain {
        let one_ic_iw_remain = one_ic_iw_remain.expect(
            &format!("unable to find iconv2d_microkernel_{}x{}", iw_block, 1)
        );
        let partial_ic_iw_remain = partial_ic_iw_remain.expect(
            &format!("unable to find oconv2d_microkernel_{}", iw_block)
        );
        with_normal_remain(
            [ii_start, ii_end],
            [in_channels, ic_block_size as i64],
            [in_width_full_end, in_width, iw_block as i64],
            [ll, l_end],
            [jj, j_end],
            [kernel_height, kernel_width],
            [isb, ish, isw],
            [osb, osh, osw],
            [step_height, step_width],
            [ph_start, pw_start],
            [dh, dw],
            [out_height, out_width],
            batch,
            remain,
            out,
            kernel,
            &inp,
            activation,
            *full_ic_kernel_iw_remain,
            one_ic_iw_remain,
            partial_ic_iw_remain
        );
    }
    *kernel += kernel_height * kernel_width * (ii_end - ii_start) * (j_end - jj);
}
