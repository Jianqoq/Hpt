use super::conv_config::Conv2dConfig;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_full_oc_kernel_dispatch;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_remain_oc_kernel_dispatch;
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
    /// * `config` - An optional reference to a `Conv2dConfig` structure that holds additional configuration parameters for optimization.
    ///   If not provided, a default configuration is used.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing the output tensor after applying the 2D convolution operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    #[inline(never)]
    pub fn iconv2d(
        &self,
        kernels: &_Tensor<T>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        _: Option<&Conv2dConfig<T>>
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

        let mut oc_nvec =
            cache_size::l1_cache_line_size().unwrap_or(crate::CACHE_LINE_SIZE) /
            core::mem::size_of::<T>() /
            T::Vec::SIZE;
        let mut ow_block = predict_ow_block(oc_nvec);
        let ic_nvec = (16).min((in_channels as usize) / T::Vec::SIZE);
        let jb = (16).min((out_channels as usize) / (T::Vec::SIZE * oc_nvec));

        // eval_micro_kernel::<T>(
        //     [ic_nvec, oc_nvec],
        //     [kernel_height as usize, kernel_width as usize],
        //     [step_height as usize, step_width as usize],
        //     [OH_BLOCK as usize, ow_block],
        //     [out_height as usize, out_width as usize],
        //     [in_channels as usize, out_channels as usize],
        //     jb
        // );

        let full_oc_kernel = iconv2d_full_oc_kernel_dispatch(&mut oc_nvec, &mut ow_block).expect(
            &format!("unable to find iconv2d_microkernel_{}x{}", ow_block, oc_nvec)
        );
        let full_oc_kernel_fn = full_oc_kernel.kernel.clone();
        let full_oc_kernel_ow_remain = iconv2d_full_oc_kernel_dispatch(
            &mut oc_nvec,
            &mut ((out_width as usize) % ow_block)
        );
        if full_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let partial_oc_kernel = iconv2d_remain_oc_kernel_dispatch(&mut ow_block);
        let partial_oc_kernel_ow_remain = iconv2d_remain_oc_kernel_dispatch(
            &mut ((out_width as usize) % ow_block)
        );
        if partial_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let num_oh = (out_height + OH_BLOCK - 1) / OH_BLOCK;
        let outer = batch * num_oh;
        let out_width_full_end = out_width - (out_width % (ow_block as i64));
        let oc_remain = out_channels % ((T::Vec::SIZE as i64) * (oc_nvec as i64));

        let ro_kernel = kernels.empty_like()?;
        let ro_ptr = ro_kernel.ptr();
        reorder_kernel(
            &kernels.ptr(),
            ro_ptr.clone(),
            jb,
            [in_channels as usize, ic_nvec],
            [out_channels as usize, oc_nvec],
            [ks0 as usize, ks1 as usize, ks2 as usize],
            [kernel_height as usize, kernel_width as usize]
        );

        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let mut kernel = ro_ptr.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * OH_BLOCK;
            let l_end = (ll + OH_BLOCK).min(out_height);
            for ii in (0..in_channels).step_by(T::Vec::SIZE * ic_nvec) {
                let i_end = (ii + (T::Vec::SIZE as i64) * (ic_nvec as i64)).min(in_channels);
                for jj in (0..out_channels).step_by(T::Vec::SIZE * oc_nvec * (jb as usize)) {
                    let jj_start = jj;
                    let jj_end = (jj + (T::Vec::SIZE as i64) * (oc_nvec as i64) * (jb as i64)).min(
                        out_channels
                    );
                    let kernel_k = kernel.clone();
                    for k in (0..out_width_full_end).step_by(ow_block) {
                        for j in (jj_start..jj_end).step_by(T::Vec::SIZE * oc_nvec) {
                            let original = kernel.clone();
                            for l in ll..l_end {
                                full_oc_kernel_fn(
                                    [ii, i_end],
                                    [kernel_height, kernel_width],
                                    [b, l, k, j],
                                    [osb, osh, osw],
                                    [step_height, step_width],
                                    [isb, ish, isw],
                                    [ph_start, pw_start],
                                    [dh, dw],
                                    &mut out,
                                    &inp,
                                    &mut kernel
                                );
                                kernel = original.clone();
                            }
                            kernel +=
                                kernel_height *
                                kernel_width *
                                (i_end - ii) *
                                (oc_nvec as i64) *
                                (T::Vec::SIZE as i64);
                        }
                        if
                            jj_end - jj_start <
                            (T::Vec::SIZE as i64) * (oc_nvec as i64) * (jb as i64)
                        {
                            if let Some(partial_oc_kernel) = partial_oc_kernel {
                                for j in (out_channels - oc_remain..out_channels).step_by(
                                    T::Vec::SIZE * oc_nvec
                                ) {
                                    let original = kernel.clone();
                                    for l in ll..l_end {
                                        partial_oc_kernel(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ph_start, pw_start],
                                            [dh, dw],
                                            oc_remain,
                                            &mut out,
                                            &inp,
                                            &mut kernel
                                        );
                                        kernel = original.clone();
                                    }
                                    kernel +=
                                        kernel_height * kernel_width * oc_remain * (i_end - ii);
                                }
                            }
                        }
                        kernel = kernel_k.clone();
                    }
                    if let Some(full_oc_kernel_ow_remain) = &full_oc_kernel_ow_remain {
                        for k in (out_width_full_end..out_width).step_by(ow_block) {
                            for j in (jj_start..jj_end).step_by(T::Vec::SIZE * oc_nvec) {
                                let original = kernel.clone();
                                for l in ll..l_end {
                                    (full_oc_kernel_ow_remain.kernel)(
                                        [ii, i_end],
                                        [kernel_height, kernel_width],
                                        [b, l, k, j],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [isb, ish, isw],
                                        [ph_start, pw_start],
                                        [dh, dw],
                                        &mut out,
                                        &inp,
                                        &mut kernel
                                    );
                                    kernel = original.clone();
                                }
                                kernel +=
                                    kernel_height *
                                    kernel_width *
                                    (i_end - ii) *
                                    (oc_nvec as i64) *
                                    (T::Vec::SIZE as i64);
                            }
                            if
                                jj_end - jj_start <
                                (T::Vec::SIZE as i64) * (oc_nvec as i64) * (jb as i64)
                            {
                                if let Some(partial_oc_ow_remain) = partial_oc_kernel_ow_remain {
                                    for j in (out_channels - oc_remain..out_channels).step_by(
                                        T::Vec::SIZE * oc_nvec
                                    ) {
                                        let original = kernel.clone();
                                        for l in ll..l_end {
                                            partial_oc_ow_remain(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ph_start, pw_start],
                                                [dh, dw],
                                                oc_remain,
                                                &mut out,
                                                &inp,
                                                &mut kernel
                                            );
                                            kernel = original.clone();
                                        }
                                        kernel +=
                                            kernel_height * kernel_width * oc_remain * (i_end - ii);
                                    }
                                }
                            }
                            kernel = kernel_k.clone();
                        }
                    }
                    kernel += kernel_height * kernel_width * (jj_end - jj_start) * (i_end - ii);
                }
            }
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
    kw: usize,
    line_size: usize
) -> usize {
    oc_nvec * ic_nvec * T::Vec::SIZE * kh * kw * jb
}

#[allow(unused)]
fn eval_micro_kernel<T: CommonBounds>(
    [ic_nvec, oc_nvec]: [usize; 2],
    [kh, kw]: [usize; 2],
    [step_height, step_width]: [usize; 2],
    [oh_block, ow_block]: [usize; 2],
    [out_height, out_width]: [usize; 2],
    [in_channels, out_channels]: [usize; 2],
    jb: usize
) {
    let (cache_line_size, l1_cache, l2_cache) = if cfg!(target_arch = "x86_64") {
        (
            cache_size::l1_cache_line_size().unwrap_or(64) / T::BIT_SIZE,
            cache_size::l1_cache_size().unwrap_or(32 * 1024) / T::BIT_SIZE,
            cache_size::l2_cache_size().unwrap_or(2 * 1024 * 1024) / T::BIT_SIZE,
        )
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        (
            128 / T::BIT_SIZE,
            cache_size::l1_cache_size().unwrap_or(32 * 1024) / T::BIT_SIZE,
            (4 * 1024 * 1024) / T::BIT_SIZE,
        ) // need to implement
    } else {
        panic!("Unsupported architecture");
    };
    let total_cache = l1_cache + l2_cache;
    // calculate when jb = 1, how much cache is used
    let inp_used = inp_used::<T>(
        oh_block,
        ow_block,
        ic_nvec,
        kh,
        kw,
        step_height,
        step_width,
        cache_line_size
    );
    let out_used = out_used::<T>(oh_block, 1, oc_nvec, ow_block, cache_line_size);
    let kernel_used = kernel_used::<T>(oc_nvec, ic_nvec, 1, kh, kw, cache_line_size);

    // calculate how many jb will fill the cache, since output and kernel will keep loading data into the cache
    let nj = if inp_used > total_cache {
        0.0
    } else {
        (((total_cache - inp_used) as f64) / ((kernel_used as f64) + (out_used as f64))).min(
            (out_channels as f64) / ((T::Vec::SIZE as f64) * (oc_nvec as f64))
        )
    };
    println!("nj: {}", nj);
    println!("jb: {}", jb);
    println!("inp_used: {}", inp_used);
    println!("out_used: {}", out_used * jb);
    println!("kernel_used: {}", kernel_used * jb);
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
    let oc_remain = out_channel % (T::Vec::SIZE * oc_nvec);
    (0..in_channel)
        .into_par_iter()
        .step_by(T::Vec::SIZE * ic_nvec)
        .for_each(|ii| {
            let i_end = (ii + T::Vec::SIZE * ic_nvec).min(in_channel);
            let mut reordered = reordered.clone() + ii * out_channel * kh * kw;
            for jj in (0..out_channel).step_by(T::Vec::SIZE * oc_nvec * jb) {
                let jj_start = jj;
                let jj_end = (jj + T::Vec::SIZE * oc_nvec * jb).min(out_channel);
                for j in (jj_start..jj_end).step_by(T::Vec::SIZE * oc_nvec) {
                    for n in 0..kh {
                        for m in 0..kw {
                            for i in ii..i_end {
                                for v in 0..oc_nvec {
                                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                                    unsafe {
                                        ptr.write_unaligned(
                                            T::Vec::from_ptr(&kernel[i * ks2 + n * ks0 + m * ks1 + j + v * T::Vec::SIZE])
                                        ); // prettier-ignore
                                    }
                                    reordered += T::Vec::SIZE;
                                }
                            }
                        }
                    }
                }
                if jj_end - jj_start < T::Vec::SIZE * oc_nvec * jb {
                    for j in (out_channel - oc_remain..out_channel).step_by(
                        T::Vec::SIZE * oc_nvec
                    ) {
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
fn find_optimal_jb<T: CommonBounds>(
    oc_nvec: usize,
    ic_nvec: usize,
    out_channels: usize,
    in_channels: usize,
    kh: usize,
    kw: usize,
    oh_block: usize,
    ow_block: usize,
    cache_line_size: usize,
    l1_cache: usize,
    l2_cache: usize
) -> usize {
    let max_jb = (out_channels as usize) / (T::Vec::SIZE * oc_nvec);
    let mut best_jb = 1;
    let mut best_score = f64::MAX;
    let total_cache = l1_cache + l2_cache;

    for jb in 1..=max_jb {
        let kernel_used = kernel_used::<T>(oc_nvec, ic_nvec, jb, kh, kw, cache_line_size);
        let inp_used = inp_used::<T>(oh_block, ow_block, ic_nvec, kh, kw, 1, 1, cache_line_size);
        let out_used = out_used::<T>(oh_block, jb, oc_nvec, ow_block, cache_line_size);

        let total_used = kernel_used + inp_used + out_used;
        
        // Calculate reuse factors
        let kernel_reuse = (out_channels * in_channels) as f64 / (jb * oc_nvec * ic_nvec * T::Vec::SIZE * T::Vec::SIZE) as f64;
        let input_reuse = out_channels as f64 / (jb * oc_nvec * T::Vec::SIZE) as f64;

        // Calculate effective cache usage
        let effective_kernel_cache = kernel_used as f64 / kernel_reuse;
        let effective_input_cache = inp_used as f64 / input_reuse;
        let effective_total_cache = effective_kernel_cache + effective_input_cache + out_used as f64;

        // Calculate cache utilization
        let cache_utilization = effective_total_cache / total_cache as f64;

        // Penalize both under-utilization and over-utilization
        let utilization_penalty = (cache_utilization - 0.8).abs();

        // Calculate parallelism score (higher is better)
        let parallelism_score = jb as f64 / max_jb as f64;

        // Combine scores (lower is better)
        let score = utilization_penalty + (1.0 - parallelism_score);

        if score < best_score {
            best_score = score;
            best_jb = jb;
        }

        println!("jb: {}, score: {:.4}, utilization: {:.2}%, parallelism: {:.2}", 
                 jb, score, cache_utilization * 100.0, parallelism_score);
    }

    println!("Best jb: {}", best_jb);
    best_jb
}