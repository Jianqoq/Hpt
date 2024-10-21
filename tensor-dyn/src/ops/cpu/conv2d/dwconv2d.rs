use crate::ops::cpu::cache_utils::cache::Cache;
use crate::ops::cpu::kernels::dwconv_kernels::bias_remain_oc_kernel_dispatch;
use crate::ops::cpu::kernels::dwconv_kernels::conv2d_full_oc_bias_kernel_dispatch;
use crate::ops::cpu::kernels::dwconv_kernels::conv2d_full_oc_kernel_dispatch;
use crate::ops::cpu::kernels::dwconv_kernels::remain_oc_kernel_dispatch;
use crate::ops::cpu::kernels::dwconv_kernels::Params;
use crate::ops::cpu::kernels::dwconv_kernels::PartialParams;
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
    bool: IntoScalar<T>,
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
    pub fn dwconv2d(
        &self,
        kernels: &_Tensor<T>,
        bias: Option<&_Tensor<T>>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(T::Vec) -> T::Vec>,
    ) -> anyhow::Result<_Tensor<T>> {
        let img_shape = self.shape();
        if img_shape.len() != 4 {
            return Err(ErrHandler::Conv2dImgShapeInCorrect(
                img_shape.len(),
                core::panic::Location::caller(),
            )
            .into());
        }
        let batch = img_shape[0];
        let img_height = img_shape[1];
        let img_width = img_shape[2];
        let in_channels = img_shape[3];
        let kernel_shape = kernels.shape();
        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];
        let k_in_channels = kernel_shape[2];
        let out_channels = kernel_shape[3];
        if 1 != k_in_channels {
            panic!("kernel in_channel must equal to 1, got {}", k_in_channels);
        }
        let (step_width, step_height) = (steps[0], steps[1]);
        let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let out_height =
            (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
        let out_width =
            (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let img = self.clone();
        if out_height <= 0 || out_width <= 0 {
            return if out_height <= 0 {
                Err(InvalidInputShape(out_height, core::panic::Location::caller()).into())
            } else {
                Err(InvalidInputShape(out_width, core::panic::Location::caller()).into())
            };
        }
        let activation = activation.unwrap_or(|x| x);
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

        let oh_block = (3).min(out_height).max(1);

        // let mut oc_nvec = cache.l1_line_size / T::Vec::SIZE;
        let mut ow_block = /*predict_ow_block(ic_nvec)*/ 5;

        // let params = kernel_params::<T>(
        //     1 as usize,
        //     1 as usize,
        //     ow_block,
        //     oc_nvec,
        //     oh_block as usize,
        //     [kernel_height as usize, kernel_width as usize],
        //     cache
        // );
        let mut ic_nvec = 8;

        // retrieve micro kernels start

        let full_oc_kernel =
            conv2d_full_oc_kernel_dispatch(&mut ic_nvec, &mut ow_block).expect(&format!(
                "unable to find iconv2d_microkernel_{}x{}",
                ow_block, ic_nvec
            ));
        let full_oc_kernel_fn = full_oc_kernel.kernel.clone();
        let full_oc_kernel_ow_remain = conv2d_full_oc_kernel_dispatch::<T>(
            &mut ic_nvec,
            &mut ((out_width as usize) % ow_block),
        );
        if full_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let partial_oc_kernel = remain_oc_kernel_dispatch::<T>(&mut ow_block);
        if let Some(partial_oc_kernel) = partial_oc_kernel {
            assert_eq!(ow_block, partial_oc_kernel.ow_block);
        }
        let partial_oc_kernel_ow_remain =
            remain_oc_kernel_dispatch::<T>(&mut ((out_width as usize) % ow_block));
        if partial_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let full_oc_kernel_fn_1_oc = conv2d_full_oc_kernel_dispatch::<T>(&mut 1, &mut ow_block);
        let full_oc_kernel_fn_1_oc_ow_remain =
            conv2d_full_oc_kernel_dispatch::<T>(&mut 1, &mut ((out_width as usize) % ow_block));
        let has_bias = bias.is_some();
        let bias_full_oc_kernel = if has_bias {
            Some(conv2d_full_oc_bias_kernel_dispatch::<T>(&mut ic_nvec, &mut ow_block).unwrap())
        } else {
            None
        };
        let bias_one_oc_kernel = if has_bias {
            Some(conv2d_full_oc_bias_kernel_dispatch::<T>(&mut 1, &mut ow_block).unwrap())
        } else {
            None
        };
        let bias_remain_oc_kernel = if has_bias {
            Some(bias_remain_oc_kernel_dispatch::<T>(&mut ow_block).unwrap())
        } else {
            None
        };
        let bias_full_oc_ow_remain = if has_bias {
            Some(
                conv2d_full_oc_bias_kernel_dispatch::<T>(
                    &mut ic_nvec,
                    &mut ((out_width as usize) % ow_block),
                )
                .unwrap(),
            )
        } else {
            None
        };
        let bias_one_oc_ow_remain = if has_bias {
            Some(
                conv2d_full_oc_bias_kernel_dispatch::<T>(
                    &mut 1,
                    &mut ((out_width as usize) % ow_block),
                )
                .unwrap(),
            )
        } else {
            None
        };
        let bias_partial_oc_ow_remain = if has_bias {
            Some(
                bias_remain_oc_kernel_dispatch::<T>(&mut ((out_width as usize) % ow_block))
                    .unwrap(),
            )
        } else {
            None
        };

        // retrieve micro kernels end

        let num_oh = (out_height + oh_block - 1) / oh_block; // div ceil, i.e. ceiling of out_height / oh_block
        let outer = batch * num_oh;
        let out_width_full_end = out_width - (out_width % (ow_block as i64)); // the end of the out width that is a multiple of ow_block

        // create new memory space to store the reordered kernel filter
        let ro_kernel = kernels.empty_like()?;
        let ro_ptr = ro_kernel.ptr();

        // reorder the kernel filter, so that when we do convolution, we can simply increment the pointer and get the data, this can significantly reduce cache miss rate and improve performance
        reorder_kernel(
            &kernels.ptr(),
            ro_ptr.clone(),
            [in_channels as usize, ic_nvec],
            [ks0 as usize, ks1 as usize],
            [kernel_height as usize, kernel_width as usize],
        );
        // println!("{}", kernels);
        // println!("{}", ro_kernel);

        let ic_block_size = ic_nvec * T::Vec::SIZE; // in channel block size
        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let mut kernel;
            let origin = ro_ptr.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * oh_block;
            let l_end = (ll + oh_block).min(out_height);
            let params = Params {
                arg1: 0,
                arg2: [kernel_height, kernel_width],
                arg3: [b, 0, 0],
                arg4: [osb, osh, osw],
                arg5: [step_height, step_width],
                arg6: [isb, ish, isw],
                pads: [ph_start, pw_start],
                arg8: [dh, dw],
                arg9: [img_height, img_width],
            };
            let full_ic_block_size_end = in_channels - (in_channels % (ic_block_size as i64));
            for ii in (0..full_ic_block_size_end).step_by(ic_block_size) {
                kernel = origin.clone() + ii * kernel_height * kernel_width;
                ow_loop(
                    [out_width_full_end, out_width],
                    [ll, l_end],
                    [ow_block],
                    &mut out,
                    &mut kernel,
                    |k, l, out, kernel| {
                        full_oc_kernel_fn(
                            Params {
                                arg1: ii,
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                    |k, l, out, kernel| {
                        (unsafe { full_oc_kernel_ow_remain.unwrap_unchecked().kernel })(
                            Params {
                                arg1: ii,
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                );
            }
            let remain = in_channels % (ic_block_size as i64);
            let remain = remain % (T::Vec::SIZE as i64);
            for ii in (full_ic_block_size_end..in_channels - remain).step_by(T::Vec::SIZE) {
                kernel = origin.clone() + ii * kernel_height * kernel_width;
                ow_loop(
                    [out_width_full_end, out_width],
                    [ll, l_end],
                    [ow_block],
                    &mut out,
                    &mut kernel,
                    |k, l, out, kernel| {
                        (unsafe { full_oc_kernel_fn_1_oc.unwrap_unchecked().kernel })(
                            Params {
                                arg1: ii,
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                    |k, l, out, kernel| {
                        (unsafe { full_oc_kernel_fn_1_oc_ow_remain.unwrap_unchecked().kernel })(
                            Params {
                                arg1: ii,
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                );
            }
            for ii in (in_channels - remain..in_channels).step_by(T::Vec::SIZE) {
                kernel = origin.clone() + ii * kernel_height * kernel_width;
                let params = PartialParams {
                    arg1: ii,
                    arg2: [kernel_height, kernel_width],
                    arg3: [b, 0, 0],
                    arg4: [osb, osh, osw],
                    arg5: [step_height, step_width],
                    arg6: [isb, ish, isw],
                    arg7: [ph_start, pw_start],
                    arg8: [dh, dw],
                    arg9: [img_height, img_width],
                    oc_remain: remain,
                };
                ow_loop(
                    [out_width_full_end, out_width],
                    [ll, l_end],
                    [ow_block],
                    &mut out,
                    &mut kernel,
                    |k, l, out, kernel| {
                        (unsafe { partial_oc_kernel.unwrap_unchecked().kernel })(
                            PartialParams {
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                    |k, l, out, kernel| {
                        (unsafe { partial_oc_kernel_ow_remain.unwrap_unchecked().kernel })(
                            PartialParams {
                                arg3: [b, l, k],
                                ..params
                            },
                            out,
                            kernel,
                            &inp,
                            activation,
                        );
                    },
                );
                kernel += kernel_height * kernel_width * remain;
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
    line_size: usize,
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
    line_size: usize,
) -> usize {
    let nv = line_size / (SIMD_WIDTH / 8 / core::mem::size_of::<T>());
    let in_range_num_w = (0..owb).take_while(|&idx| idx * step_width < kw).count();
    let in_range_num_h = (0..lb).take_while(|&idx| idx * step_height < kh).count();
    owb * ic_nvec.div_ceil(nv)
        * (kh + lb - in_range_num_h)
        * (kw + owb - in_range_num_w)
        * line_size
}
#[allow(unused)]
fn kernel_used<T: CommonBounds>(
    oc_nvec: usize,
    ic_nvec: usize,
    jb: usize,
    kh: usize,
    kw: usize,
) -> usize {
    oc_nvec * ic_nvec * T::Vec::SIZE * kh * kw * jb
}

fn reorder_kernel<T: CommonBounds>(
    kernel: &Pointer<T>,
    reordered: Pointer<T>,
    [in_channel, ic_nvec]: [usize; 2],
    [ks0, ks1]: [usize; 2],
    [kh, kw]: [usize; 2],
) {
    let ic_block_size = ic_nvec * T::Vec::SIZE;
    let full_ic_block_size_end = in_channel - (in_channel % ic_block_size);
    (0..full_ic_block_size_end)
        .into_par_iter()
        .step_by(ic_block_size)
        .for_each(|ii| {
            let mut reordered = reordered.clone() + ii * kh * kw;
            for n in 0..kh {
                for m in 0..kw {
                    for i in 0..ic_nvec {
                        let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                        let inp = unsafe {
                            T::Vec::from_ptr(&kernel[ii + i * T::Vec::SIZE + n * ks0 + m * ks1])
                        };
                        unsafe {
                            ptr.write_unaligned(inp);
                        }
                        reordered += T::Vec::SIZE;
                    }
                }
            }
        });
    let remain = in_channel % ic_block_size;
    let remain = remain % T::Vec::SIZE;
    (full_ic_block_size_end..in_channel - remain)
        .into_par_iter()
        .step_by(T::Vec::SIZE)
        .for_each(|ii| {
            let mut reordered = reordered.clone() + ii * kh * kw;
            for n in 0..kh {
                for m in 0..kw {
                    let ptr = reordered.ptr as *mut _ as *mut T::Vec;
                    let inp = unsafe { T::Vec::from_ptr(&kernel[ii + n * ks0 + m * ks1]) };
                    unsafe {
                        ptr.write_unaligned(inp);
                    }
                    reordered += T::Vec::SIZE;
                }
            }
        });
    (in_channel - remain..in_channel)
        .into_par_iter()
        .step_by(T::Vec::SIZE)
        .for_each(|ii| {
            let mut reordered = reordered.clone() + ii * kh * kw;
            for n in 0..kh {
                for m in 0..kw {
                    for i in 0..remain {
                        let ptr = reordered.ptr;
                        let inp = kernel[ii + n * ks0 + m * ks1 + i];
                        unsafe {
                            ptr.write(inp);
                        }
                        reordered += 1usize;
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
    cache: Cache<T>,
) -> (usize, usize) {
    let l1 = cache.l1;
    let l2 = cache.l2;

    let ic_range = 1..(in_channels as usize) / T::Vec::SIZE;
    let jb_range = 1..(out_channels as usize) / (T::Vec::SIZE * oc_nvec);

    let best_params = ic_range
        .into_par_iter()
        .flat_map(|ic| jb_range.clone().into_par_iter().map(move |jb| (ic, jb)))
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
            },
        );

    (best_params.0, best_params.1)
}

fn ow_loop<F1, F2, T: CommonBounds>(
    [out_width_full_end, out_width]: [i64; 2],
    [ll, l_end]: [i64; 2],
    [ow_block]: [usize; 1],
    out: &mut Pointer<T>,
    kernel: &mut Pointer<T>,
    full_oc_kernel: F1,
    full_oc_kernel_ow_remain: F2,
) where
    F1: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
    F2: Fn(i64, i64, &mut Pointer<T>, &mut Pointer<T>),
{
    for k in (0..out_width_full_end).step_by(ow_block) {
        let original = kernel.clone();
        for l in ll..l_end {
            full_oc_kernel(k, l, out, kernel);
            *kernel = original.clone();
        }
    }
    if out_width > out_width_full_end {
        for k in (out_width_full_end..out_width).step_by(ow_block) {
            let original = kernel.clone();
            for l in ll..l_end {
                full_oc_kernel_ow_remain(k, l, out, kernel);
                *kernel = original.clone();
            }
        }
    }
}
