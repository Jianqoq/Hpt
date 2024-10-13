use crate::ops::cpu::cache_utils::cache::Cache;
use crate::ops::cpu::kernels::conv_kernels::conv2d_full_oc_bias_kernel_dispatch;
use crate::ops::cpu::kernels::conv_kernels::conv2d_full_oc_kernel_dispatch;
use crate::ops::cpu::kernels::conv_kernels::conv2d_remain_oc_kernel_dispatch;
use crate::ops::cpu::kernels::conv_kernels::ConvKernel;
use crate::ops::cpu::kernels::conv_kernels::ConvPartialKernel;
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
        T::Vec: VecTrait<T> + Copy + Init<T> + Send + Sync + VecCommon + NormalOut<Output = T::Vec>
{
    /// Performs a 2D convolution operation on the input tensor.
    ///
    /// This method applies a 2D convolution operation on the tensor using the specified kernel,
    /// strides (steps), padding, and dilation factors. It optionally accepts a configuration (`Conv2dConfig`)
    /// to fine-tune the performance, such as optimizing for cache usage and block sizes.
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
    pub fn conv2d(
        &self,
        kernels: &_Tensor<T>,
        bias: Option<&_Tensor<T>>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2]
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
        let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let out_height =
            (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
        let out_width =
            (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
            self.pad(
                &[
                    (0, 0),
                    (ph_start, ph_end),
                    (pw_start, pw_end),
                    (0, 0),
                ],
                T::ZERO
            )?
        } else {
            self.clone()
        };
        if out_height <= 0 || out_width <= 0 {
            return if out_height <= 0 {
                Err(InvalidInputShape(out_height, core::panic::Location::caller()).into())
            } else {
                Err(InvalidInputShape(out_width, core::panic::Location::caller()).into())
            };
        }
        let output = _Tensor::<T>::empty([batch, out_height, out_width, out_channels])?;
        let out = output.ptr();
        let inp = img.ptr();

        let osb = output.strides()[0]; // batch
        let osh = output.strides()[1]; // height
        let osw = output.strides()[2]; // width

        let isb = img.strides()[0]; // batch
        let ish = img.strides()[1]; // height
        let isw = img.strides()[2]; // width

        let ks0 = kernels.strides()[0]; // kernel_height
        let ks1 = kernels.strides()[1]; // kernel_width
        let ks2 = kernels.strides()[2]; // in_channels

        const OH_BLOCK: i64 = 3;

        let cache = Cache::<T>::new();

        let mut oc_nvec = cache.l1_line_size / T::Vec::SIZE;
        let mut ow_block = predict_ow_block(oc_nvec);

        let params = kernel_params::<T>(
            out_channels as usize,
            in_channels as usize,
            ow_block,
            oc_nvec,
            OH_BLOCK as usize,
            [kernel_height as usize, kernel_width as usize],
            cache
        );
        let (ic_nvec, jb) = params;
        let full_oc_kernel = conv2d_full_oc_kernel_dispatch(&mut oc_nvec, &mut ow_block).expect(
            &format!("unable to find iconv2d_microkernel_{}x{}", ow_block, oc_nvec)
        ); // get the micro kernel for the full part of out width and full part of out channel
        let full_oc_kernel_fn = full_oc_kernel.kernel.clone();
        let full_oc_kernel_ow_remain = conv2d_full_oc_kernel_dispatch::<T>(
            &mut oc_nvec,
            &mut ((out_width as usize) % ow_block)
        ); // get the micro kernel for the remain part of out width that is not a multiple of ow_block, and full part of out channel
        if full_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let partial_oc_kernel = conv2d_remain_oc_kernel_dispatch::<T>(&mut ow_block);
        if let Some(partial_oc_kernel) = partial_oc_kernel {
            assert_eq!(ow_block, partial_oc_kernel.ow_block);
        }
        let partial_oc_kernel_ow_remain = conv2d_remain_oc_kernel_dispatch::<T>(
            &mut ((out_width as usize) % ow_block)
        );
        if partial_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let full_oc_kernel_fn_1_oc = conv2d_full_oc_kernel_dispatch::<T>(&mut 1, &mut ow_block);
        let full_oc_kernel_fn_1_oc_ow_remain = conv2d_full_oc_kernel_dispatch::<T>(
            &mut 1,
            &mut ((out_width as usize) % ow_block)
        );
        let num_oh = (out_height + OH_BLOCK - 1) / OH_BLOCK; // div ceil, i.e. ceiling of out_height / OH_BLOCK
        let outer = batch * num_oh;
        let out_width_full_end = out_width - (out_width % (ow_block as i64)); // the end of the out width that is a multiple of ow_block

        // create new memory space to store the reordered kernel filter
        let ro_kernel = kernels.empty_like()?;
        let ro_ptr = ro_kernel.ptr();

        // reorder the kernel filter, so that when we do convolution, we can simply increment the pointer and get the data, this can significantly reduce cache miss rate and improve performance
        reorder_kernel(
            &kernels.ptr(),
            ro_ptr.clone(),
            jb,
            [in_channels as usize, ic_nvec],
            [out_channels as usize, oc_nvec],
            [ks0 as usize, ks1 as usize, ks2 as usize],
            [kernel_height as usize, kernel_width as usize]
        );

        let ic_block_size = ic_nvec * T::Vec::SIZE; // in channel block size
        let oc_block_size = oc_nvec * T::Vec::SIZE; // out channel block size, but this is only caculated based on cache line size, we have another block size `jb`

        let has_bias = bias.is_some();
        let bias_full_oc_kernel = if has_bias {
            Some(conv2d_full_oc_bias_kernel_dispatch::<T>(&mut 1, &mut oc_nvec).unwrap())
        } else {
            None
        };
        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let mut kernel = ro_ptr.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * OH_BLOCK;
            let l_end = (ll + OH_BLOCK).min(out_height);
            match (partial_oc_kernel, full_oc_kernel_ow_remain, partial_oc_kernel_ow_remain) {
                (None, None, None) => {
                    for ii in (0..in_channels).step_by(ic_block_size) {
                        let i_end = (ii + (ic_block_size as i64)).min(in_channels);
                        let bias_kernel = i_end == in_channels && has_bias;

                        if bias_kernel {
                            let bias = bias.unwrap().ptr();
                            let bias_full_oc_kernel = bias_full_oc_kernel.unwrap();
                            conv_perfect(
                                out_channels,
                                oc_block_size,
                                ow_block,
                                out_width_full_end,
                                [kernel_height, kernel_width],
                                [ll, l_end],
                                i_end - ii,
                                jb,
                                &mut kernel,
                                &mut out,
                                |l, k, j, kernel, out| {
                                    bias_full_oc_kernel(
                                        [ii, i_end],
                                        [kernel_height, kernel_width],
                                        [b, l, k, j],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [isb, ish, isw],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        &bias,
                                        out,
                                        &inp,
                                        kernel
                                    );
                                }
                            );
                        } else {
                            conv_perfect(
                                out_channels,
                                oc_block_size,
                                ow_block,
                                out_width_full_end,
                                [kernel_height, kernel_width],
                                [ll, l_end],
                                i_end - ii,
                                jb,
                                &mut kernel,
                                &mut out,
                                |l, k, j, kernel, out| {
                                    full_oc_kernel_fn(
                                        [ii, i_end],
                                        [kernel_height, kernel_width],
                                        [b, l, k, j],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [isb, ish, isw],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        out,
                                        &inp,
                                        kernel
                                    );
                                }
                            );
                        }
                    }
                }
                _ => {
                    for ii in (0..in_channels).step_by(ic_block_size) {
                        let i_end = (ii + (ic_block_size as i64)).min(in_channels);
                        // out channel has two levels of blocking:
                        // 1. it first blocks by oc_block_size * jb
                        // 2. it then blocks by oc_block_size (cache line size)
                        for jj in (0..out_channels).step_by(oc_block_size * (jb as usize)) {
                            let jj_start = jj;

                            // make sure jj_end is in the range of out_channels
                            let jj_end = (jj + (oc_block_size as i64) * (jb as i64)).min(
                                out_channels
                            );

                            // calculate the remain part that are less than T::Vec::SIZE * oc_nvec
                            let remain = (jj_end - jj_start) % (oc_block_size as i64);
                            let kernel_k = kernel.clone();
                            // handle the out width full part
                            for k in (0..out_width_full_end).step_by(ow_block) {
                                func(
                                    jj_start,
                                    jj_end,
                                    remain,
                                    oc_block_size,
                                    &mut kernel,
                                    full_oc_kernel_fn,
                                    &full_oc_kernel_fn_1_oc,
                                    &partial_oc_kernel,
                                    [ii, i_end],
                                    [kernel_height, kernel_width],
                                    [b, k],
                                    [osb, osh, osw],
                                    [step_height, step_width],
                                    [isb, ish, isw],
                                    [ph_start, pw_start],
                                    [dh, dw],
                                    &mut out,
                                    &inp,
                                    ll,
                                    l_end,
                                    out_channels
                                );
                                kernel = kernel_k.clone();
                            }
                            // handle the out width remain part
                            if let Some(full_oc_kernel_ow_remain) = &full_oc_kernel_ow_remain {
                                for k in (out_width_full_end..out_width).step_by(ow_block) {
                                    func(
                                        jj_start,
                                        jj_end,
                                        remain,
                                        oc_block_size,
                                        &mut kernel,
                                        full_oc_kernel_ow_remain.kernel,
                                        &full_oc_kernel_fn_1_oc_ow_remain,
                                        &partial_oc_kernel_ow_remain,
                                        [ii, i_end],
                                        [kernel_height, kernel_width],
                                        [b, k],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [isb, ish, isw],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        &mut out,
                                        &inp,
                                        ll,
                                        l_end,
                                        out_channels
                                    );
                                    kernel = kernel_k.clone();
                                }
                            }
                            kernel +=
                                kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
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
    [in_channel, ic_nvec]: [usize; 2],
    [out_channel, oc_nvec]: [usize; 2],
    [ks0, ks1, ks2]: [usize; 3],
    [kh, kw]: [usize; 2]
) {
    (0..in_channel)
        .into_par_iter()
        .step_by(T::Vec::SIZE * ic_nvec)
        .for_each(|ii| {
            let i_end = (ii + T::Vec::SIZE * ic_nvec).min(in_channel);
            let mut reordered = reordered.clone() + ii * out_channel * kh * kw;
            for jj in (0..out_channel).step_by(T::Vec::SIZE * oc_nvec * jb) {
                let jj_start = jj;
                let jj_end = (jj + T::Vec::SIZE * oc_nvec * jb).min(out_channel);
                let remain = (jj_end - jj_start) % (T::Vec::SIZE * oc_nvec);
                let oc_remain = remain % T::Vec::SIZE;
                for j in (jj_start..jj_end - remain).step_by(T::Vec::SIZE * oc_nvec) {
                    for n in 0..kh {
                        for m in 0..kw {
                            for i in ii..i_end {
                                for v in 0..oc_nvec {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(T::Vec::from_ptr(
                                            &kernel[i * ks2
                                                + n * ks0
                                                + m * ks1
                                                + j
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
                    for j in (out_channel - remain..out_channel - oc_remain).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for i in ii..i_end {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(
                                            T::Vec::from_ptr(
                                                &kernel[n * ks0 + m * ks1 + i * ks2 + j]
                                            )
                                        );
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                    for j in (out_channel - oc_remain..out_channel).step_by(T::Vec::SIZE) {
                        for n in 0..kh {
                            for m in 0..kw {
                                for i in ii..i_end {
                                    let ptr: *mut T = reordered.ptr;
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            &kernel[n * ks0 + m * ks1 + i * ks2 + j] as *const T,
                                            ptr,
                                            oc_remain
                                        );
                                    }
                                    reordered += oc_remain;
                                }
                            }
                        }
                    }
                }
            }
        });
}

fn predict_ow_block(oc_block: usize) -> usize {
    REGNUM / (oc_block + 1)
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

fn func<T: CommonBounds>(
    jj_start: i64,
    jj_end: i64,
    remain: i64,
    oc_block_size: usize,
    kernel: &mut Pointer<T>,
    full_oc: fn(
        [i64; 2],
        [i64; 2],
        [i64; 4],
        [i64; 3],
        [i64; 2],
        [i64; 3],
        [i64; 2],
        [i64; 2],
        &mut Pointer<T>,
        &Pointer<T>,
        &mut Pointer<T>
    ),
    one_oc: &Option<ConvKernel<T>>,
    partial_oc: &Option<ConvPartialKernel<T>>,
    [ii, i_end]: [i64; 2],
    [kernel_height, kernel_width]: [i64; 2],
    [b, k]: [i64; 2],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [ph_start, pw_start]: [i64; 2],
    [dh, dw]: [i64; 2],
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    ll: i64,
    l_end: i64,
    out_channels: i64
) {
    for j in (jj_start..jj_end - remain).step_by(oc_block_size) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_oc(
                [ii, i_end],
                [kernel_height, kernel_width],
                [b, l, k, j],
                [osb, osh, osw],
                [step_height, step_width],
                [isb, ish, isw],
                [ph_start, pw_start],
                [dh, dw],
                out,
                inp,
                kernel
            );
            *kernel = original.clone();
        }
        *kernel += kernel_height * kernel_width * (i_end - ii) * (oc_block_size as i64);
    }
    if remain > 0 {
        let oc_remain = remain % (T::Vec::SIZE as i64);
        // loop over the remain part that are multiple of T::Vec::SIZE
        if let Some(one_oc) = &one_oc {
            for j in (out_channels - remain..out_channels - oc_remain).step_by(T::Vec::SIZE) {
                let original = kernel.clone();
                for l in ll..l_end {
                    (one_oc.kernel)(
                        [ii, i_end],
                        [kernel_height, kernel_width],
                        [b, l, k, j],
                        [osb, osh, osw],
                        [step_height, step_width],
                        [isb, ish, isw],
                        [ph_start, pw_start],
                        [dh, dw],
                        out,
                        inp,
                        kernel
                    );
                    *kernel = original.clone();
                }
                *kernel += kernel_height * kernel_width * (T::Vec::SIZE as i64) * (i_end - ii);
            }
        }
        let partial_oc = partial_oc.expect("unable to find partial_oc_kernel").kernel;
        // loop over the remain part that are less than T::Vec::SIZE
        for j in (out_channels - oc_remain..out_channels).step_by(T::Vec::SIZE) {
            let original = kernel.clone();
            for l in ll..l_end {
                partial_oc(
                    [ii, i_end],
                    [kernel_height, kernel_width],
                    [b, l, k, j],
                    [osb, osh, osw],
                    [step_height, step_width],
                    [isb, ish, isw],
                    [ph_start, pw_start],
                    [dh, dw],
                    oc_remain,
                    out,
                    inp,
                    kernel
                );
                *kernel = original.clone();
            }
            *kernel += kernel_height * kernel_width * oc_remain * (i_end - ii);
        }
    }
}

fn conv_perfect<T: CommonBounds, F>(
    out_channels: i64,
    oc_block_size: usize,
    ow_block: usize,
    out_width_full_end: i64,
    [kh, kw]: [i64; 2],
    [ll, l_end]: [i64; 2],
    i_range: i64,
    jb: usize,
    kernel: &mut Pointer<T>,
    out: &mut Pointer<T>,
    kernel_func: F
)
    where F: Fn(i64, i64, i64, &mut Pointer<T>, &mut Pointer<T>)
{
    // out channel has two levels of blocking:
    // 1. it first blocks by oc_block_size * jb
    // 2. it then blocks by oc_block_size (cache line size)
    for jj in (0..out_channels).step_by(oc_block_size * (jb as usize)) {
        let jj_start = jj;

        // make sure jj_end is in the range of out_channels
        let jj_end = (jj + (oc_block_size as i64) * (jb as i64)).min(out_channels);

        // the kernel filter has nothing to do with out height, hence, when iterate over (0..out_width_full_end), kernel pointer should reset in each iteration
        let kernel_k = kernel.clone();

        for k in (0..out_width_full_end).step_by(ow_block) {
            for j in (jj_start..jj_end).step_by(oc_block_size) {
                // the kernel filter has nothing to do with out height, hence, when iterate over (l..l_end), kernel pointer should reset in each iteration
                let original = kernel.clone();
                for l in ll..l_end {
                    // execute micro kernel, see `iconv2d_full_oc_kernel_dispatch` for more details
                    kernel_func(l, k, j, kernel, out);
                    *kernel = original.clone();
                }
                // update the kernel pointer
                *kernel += kh * kw * i_range * (oc_block_size as i64);
            }
            *kernel = kernel_k.clone();
        }

        *kernel += kh * kw * (jj_end - jj_start) * i_range;
    }
}
