use crate::ops::cpu::conv_config::KernelParamAlgo;
use crate::ops::cpu::kernels::maxpool_kernels::*;
use crate::tensor_base::_Tensor;
use crate::CONV_REGNUM;
use rayon::prelude::*;
use tensor_common::err_handler::ErrHandler::InvalidCacheParam;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
use tensor_common::pointer::Pointer;
use tensor_traits::CommonBounds;
use tensor_traits::TensorCreator;
use tensor_traits::TensorInfo;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::NormalOut;
use tensor_types::vectors::traits::*;

use super::conv_config::Conv2dConfig;

fn case1_helper<T, const REGNUM: usize>(
    [kh, kw]: [i64; 2],
    [b, l, c]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_width, step_height]: [i64; 2],
    [dh, dw]: [i64; 2],
    co_b: i64,
    [num_wo_b, num_co_rb]: [i64; 2],
    inp_cpy: &Pointer<T>,
    mut out: Pointer<T>,
    micro_kernel: fn(i64, i64, i64, i64, &Pointer<T>, &mut [T::Vec; REGNUM]),
) where
    T: CommonBounds + IntoScalar<T>,
{
    for j in 0..num_co_rb {
        let mut res_buffer = [T::Vec::splat(T::ZERO); REGNUM];
        for n in 0..kh {
            for m in 0..kw {
                micro_kernel(
                    num_wo_b,
                    b * isb
                        + (l * step_height + n * dh) * ish
                        + m * dw * isw
                        + c * co_b
                        + j * (T::Vec::SIZE as i64), // prettier-ignore
                    step_width,
                    isw,
                    &inp_cpy,
                    &mut res_buffer,
                );
            }
        }
        for h in 0..REGNUM as i64 {
            let out_vec = &mut out[c * co_b
                + b * osb
                + l * osh
                + (num_wo_b * (CONV_REGNUM as i64) + h) * osw
                + j * (T::Vec::SIZE as i64)] as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(res_buffer[h as usize]);
            }
        }
    }
}

fn case1_remain1_helper<T, const REGNUM: usize>(
    [kh, kw]: [i64; 2],
    [b, l, c]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_width, step_height]: [i64; 2],
    [dh, dw]: [i64; 2],
    co_b: i64,
    num_wo_b: i64,
    inp_cpy: &Pointer<T>,
    mut out: Pointer<T>,
    micro_kernel: fn(i64, i64, i64, i64, &Pointer<T>, &mut [T; REGNUM]),
) where
    T: CommonBounds + IntoScalar<T> + NormalOut<Output = T>,
{
    let mut res_buffer = [T::ZERO; REGNUM];
    for n in 0..kh {
        for m in 0..kw {
            micro_kernel(
                num_wo_b,
                b * isb + (l * step_height + n * dh) * ish + m * dw * isw + c * co_b,
                step_width,
                isw,
                &inp_cpy,
                &mut res_buffer,
            );
        }
    }
    for h in 0..REGNUM as i64 {
        let out_vec =
            &mut out[c * co_b + b * osb + l * osh + (num_wo_b * (CONV_REGNUM as i64) + h) * osw];
        *out_vec = res_buffer[h as usize];
    }
}

fn case3_helper<T, const REGNUM: usize>(
    [kh, kw]: [i64; 2],
    [b, l, c]: [i64; 3],
    [isb, ish, isw]: [i64; 3],
    [osb, osh, osw]: [i64; 3],
    [step_width, step_height]: [i64; 2],
    [dh, dw]: [i64; 2],
    co_b: i64,
    num_wo_b: i64,
    co_b_remain: i64,
    wo_b_remain: i64,
    inp: &Pointer<T>,
    store_fn: fn(i64, i64, i64, i64, &mut Vec<Vec<T::Vec>>, &mut Pointer<T>),
    fast_micro_kernel: fn(i64, i64, i64, i64, &Pointer<T>, &mut [T::Vec]),
    micro_kernel: fn(i64, i64, i64, i64, i64, &Pointer<T>, &mut Vec<Vec<T::Vec>>),
    mut out: &mut Pointer<T>,
) where
    T: CommonBounds + IntoScalar<T> + NormalOut<Output = T>,
{
    let num_vec_size = co_b_remain / (T::Vec::SIZE as i64);
    let remain = co_b_remain % (T::Vec::SIZE as i64);
    if remain == 0 {
        for j in 0..num_vec_size {
            let mut res_buffer = [T::Vec::splat(T::ZERO); REGNUM];
            for n in 0..kh {
                for m in 0..kw {
                    fast_micro_kernel(
                        num_wo_b,
                        b * isb
                            + (l * step_height + n * dh) * ish
                            + m * dw * isw
                            + c * co_b
                            + j * (T::Vec::SIZE as i64), // prettier-ignore
                        step_width,
                        isw,
                        &inp,
                        &mut res_buffer,
                    );
                }
            }
            for h in 0..REGNUM as i64 {
                let out_vec = &mut out[c * co_b
                    + b * osb
                    + l * osh
                    + (num_wo_b * (CONV_REGNUM as i64) + h) * osw
                    + j * T::Vec::SIZE as i64] as *mut _
                    as *mut T::Vec; // prettier-ignore
                unsafe {
                    *out_vec = res_buffer[h as usize];
                }
            }
        }
    } else {
        let mut remain_buffer =
            vec![vec![T::Vec::splat(T::ZERO); wo_b_remain as usize]; num_vec_size as usize + 1];
        for n in 0..kh {
            for m in 0..kw {
                micro_kernel(
                    num_vec_size,
                    num_wo_b,
                    b * isb + (l * step_height + n * dh) * ish + m * dw * isw + c * co_b,
                    step_width,
                    isw,
                    &inp,
                    &mut remain_buffer,
                );
            }
        }
        for j in 0..num_vec_size {
            let buffers = &mut remain_buffer[j as usize];
            for r in 0..REGNUM as i64 {
                buffers[r as usize] = buffers[r as usize];
            }
        }
        let buffers = &mut remain_buffer[num_vec_size as usize];
        for r in 0..REGNUM {
            let buffer = &mut buffers[r];
            for i in 0..remain {
                buffer[i as usize] = buffer[i as usize];
            }
        }
        store_fn(
            num_vec_size,
            remain,
            osw,
            c * co_b + b * osb + l * osh + num_wo_b * (CONV_REGNUM as i64) * osw,
            &mut remain_buffer,
            &mut out,
        );
    }
}

impl<T> _Tensor<T>
where
    T: CommonBounds + IntoScalar<T> + NormalOut<Output = T>,
    i64: IntoScalar<T>,
    T::Vec: VecTrait<T> + Copy + Init<T> + Send + Sync + VecCommon + NormalOut<Output = T::Vec>,
{
    /// Applies 2D max pooling to the input tensor.
    ///
    /// This method performs max pooling, which reduces the spatial dimensions of the input tensor by
    /// selecting the maximum value from each region defined by the `kernel_shape`. Max pooling is commonly
    /// used in convolutional neural networks to downsample the input, reduce the number of parameters,
    /// and control overfitting.
    ///
    /// # Arguments
    ///
    /// * `kernel_shape` - A 2-element array specifying the height and width of the pooling kernel.
    ///   This defines the size of the region over which the maximum value is computed.
    /// * `steps` - A 2-element array specifying the stride (step size) of the pooling operation
    ///   along the height and width dimensions. The stride determines how far the pooling window moves
    ///   in each step.
    /// * `padding` - A 2-element array of tuples specifying the amount of padding added to the input
    ///   tensor along the height and width dimensions. Each tuple contains the padding before and after
    ///   the region along the respective axis.
    /// * `dilation` - A 2-element array specifying the dilation factor for the pooling operation. Dilation
    ///   allows the pooling window to cover a larger area by spacing out the elements being pooled.
    /// * `config` - An optional reference to a `Conv2dConfig` structure that holds configuration parameters
    ///   for optimizing the pooling operation. If not provided, default settings are used.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a new tensor with the result of the max pooling operation.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn max_pool2d(
        &self,
        kernel_shape: [i64; 2],
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        config: Option<&Conv2dConfig<T>>,
    ) -> anyhow::Result<_Tensor<T>> {
        use crate::CONV_REGNUM;

        let img_shape = self.shape();
        let batch = img_shape[0];
        let img_height = img_shape[1];
        let img_width = img_shape[2];
        let out_channels = img_shape[3];
        let kernel_height = kernel_shape[0];
        let kernel_width = kernel_shape[1];
        let (step_width, step_height) = (steps[0], steps[1]);
        let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let out_height =
            (img_height + ph_start + ph_end - dh * (kernel_height - 1) - 1) / step_height + 1;
        let out_width =
            (img_width + pw_start + pw_end - dw * (kernel_width - 1) - 1) / step_width + 1;
        let img = if !padding.iter().all(|(a, b)| *a == 0 && *b == 0) {
            self.pad(
                &[(0, 0), (ph_start, ph_end), (pw_start, pw_end), (0, 0)],
                T::ZERO,
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

        let (_, co_b) = match config {
            Some(config) => (config.ci_block_size, config.co_block_size),
            None => {
                let config = {
                    Conv2dConfig::<T>::new(
                        out_channels,
                        1,
                        [kernel_height, kernel_width],
                        KernelParamAlgo::Greedy,
                    )
                };
                (config.ci_block_size, config.co_block_size)
            }
        };
        let num_co_b = out_channels / co_b;
        let num_wo_b = out_width / (CONV_REGNUM as i64);

        let co_b_remain = out_channels % co_b;
        let wo_b_remain = out_width % (CONV_REGNUM as i64);
        let num_co_rb = co_b / (T::Vec::SIZE as i64);
        if !(co_b % (T::Vec::SIZE as i64) == 0 || co_b == 1) || co_b > out_channels {
            return Err(InvalidCacheParam(
                "co_b",
                out_channels,
                T::Vec::SIZE as i64,
                co_b,
                core::panic::Location::caller(),
            )
            .into());
        }
        let num_vec_size = co_b_remain / (T::Vec::SIZE as i64);
        let outer = batch * num_co_b * out_height;

        let inp_cpy = inp.clone();

        let case0_init = move |b: i64,
                               l: i64,
                               c: i64,
                               inner_size: i64,
                               micro_kernel_fn: fn(
            i64,
            i64,
            i64,
            i64,
            &Pointer<T>,
            &mut [T::Vec; CONV_REGNUM],
        ),
                               mut out: Pointer<T>| {
            for kp in 0..num_wo_b {
                for j in 0..inner_size {
                    let mut res_buffer = [T::Vec::splat(T::ZERO); CONV_REGNUM];
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            micro_kernel_fn(
                                kp,
                                b * isb
                                    + (l * step_height + n * dh) * ish
                                    + m * dw * isw
                                    + c * co_b
                                    + j * (T::Vec::SIZE as i64), // prettier-ignore
                                step_width,
                                isw,
                                &inp_cpy,
                                &mut res_buffer,
                            );
                        }
                    }
                    for h in 0..CONV_REGNUM as i64 {
                        unsafe {
                            let out_vec = &mut out[c * co_b
                                + b * osb
                                + l * osh
                                + (kp * (CONV_REGNUM as i64) + h) * osw
                                + j * (T::Vec::SIZE as i64)]
                                as *mut _ as *mut T::Vec; // prettier-ignore
                            out_vec.write_unaligned(res_buffer[h as usize]);
                        }
                    }
                }
            }
        };

        let case0 = move |b: i64,
                          l: i64,
                          c: i64,
                          inner_size: i64,
                          micro_kernel_fn: fn(
            i64,
            i64,
            i64,
            i64,
            &Pointer<T>,
            &mut [T::Vec; CONV_REGNUM],
        ),
                          out: Pointer<T>| {
            case0_init(b, l, c, inner_size, micro_kernel_fn, out);
        };
        let inp_cpy = inp.clone();

        let case0_remain1 = move |b: i64, l: i64, c: i64, mut out: Pointer<T>| {
            for kp in 0..num_wo_b {
                let mut res_buffer = [<T>::ZERO; CONV_REGNUM];
                for n in 0..kernel_height {
                    for m in 0..kernel_width {
                        micro_kernel_regnum_1::<T>(
                            kp,
                            b * isb + (l * step_height + n * dh) * ish + m * dw * isw + c * co_b,
                            step_width,
                            isw,
                            &inp_cpy,
                            &mut res_buffer,
                        );
                    }
                }
                for h in 0..CONV_REGNUM as i64 {
                    let out_vec = &mut out
                        [c * co_b + b * osb + l * osh + (kp * (CONV_REGNUM as i64) + h) * osw]; // prettier-ignore
                    *out_vec = res_buffer[h as usize];
                }
            }
        };

        let inp_cpy = inp.clone();
        let case1 = move |b: i64, l: i64, c: i64, out: Pointer<T>| match wo_b_remain {
            1 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_1::<T, 1>,
                );
            }
            2 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_2::<T, 2>,
                );
            }
            3 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_3::<T, 3>,
                );
            }
            4 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_4::<T, 4>,
                );
            }
            5 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_5::<T, 5>,
                );
            }
            6 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_6::<T, 6>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            7 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_7::<T, 7>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            8 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_8::<T, 8>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            9 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_9::<T, 9>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            10 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_10::<T, 10>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            11 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_11::<T, 11>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            12 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_12::<T, 12>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            13 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_13::<T, 13>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            14 => {
                case1_helper(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    [num_wo_b, num_co_rb],
                    &inp_cpy,
                    out,
                    micro_kernel_14::<T, 14>,
                );
            }
            _ => unimplemented!(),
        };

        let inp_cpy = inp.clone();
        let case1_remain1 = move |b: i64, l: i64, c: i64, out: Pointer<T>| match wo_b_remain {
            1 => {
                case1_remain1_helper::<T, 1>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_1_1::<T>,
                );
            }
            2 => {
                case1_remain1_helper::<T, 2>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_2_1::<T>,
                );
            }
            3 => {
                case1_remain1_helper::<T, 3>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_3_1::<T>,
                );
            }
            4 => {
                case1_remain1_helper::<T, 4>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_4_1::<T>,
                );
            }
            5 => {
                case1_remain1_helper::<T, 5>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_5_1::<T>,
                );
            }
            6 => {
                case1_remain1_helper::<T, 6>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_6_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            7 => {
                case1_remain1_helper::<T, 7>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_7_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            8 => {
                case1_remain1_helper::<T, 8>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_8_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            9 => {
                case1_remain1_helper::<T, 9>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_9_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            10 => {
                case1_remain1_helper::<T, 10>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_10_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            11 => {
                case1_remain1_helper::<T, 11>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_11_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            12 => {
                case1_remain1_helper::<T, 12>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_12_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            13 => {
                case1_remain1_helper::<T, 13>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_13_1::<T>,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            14 => {
                case1_remain1_helper::<T, 14>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    &inp_cpy,
                    out,
                    micro_kernel_14_1::<T>,
                );
            }
            _ => unimplemented!(),
        };

        let inp_cpy = inp.clone();

        let case0 = &case0;
        let case2 =
            move |b: i64, l: i64, c: i64, num_vec_size: i64, remain: i64, mut out: Pointer<T>| {
                let mut res_buffer =
                    vec![vec![T::Vec::splat(T::ZERO); CONV_REGNUM]; num_vec_size as usize + 1];
                for kp in 0..num_wo_b {
                    res_buffer.iter_mut().for_each(|x| {
                        x.iter_mut().for_each(|y| {
                            *y = T::Vec::splat(T::ZERO);
                        })
                    });
                    for n in 0..kernel_height {
                        for m in 0..kernel_width {
                            micro_kernel_regnum_with_buffer::<T>(
                                num_vec_size,
                                kp,
                                b * isb
                                    + (l * step_height + n * dh) * ish
                                    + m * dw * isw
                                    + c * co_b,
                                step_width,
                                isw,
                                &inp_cpy,
                                &mut res_buffer,
                            );
                        }
                    }
                    for j in 0..num_vec_size {
                        let buffers = &mut res_buffer[j as usize];
                        for r in 0..CONV_REGNUM as i64 {
                            buffers[r as usize] = buffers[r as usize];
                        }
                    }
                    for r in 0..CONV_REGNUM {
                        let buffer = &mut res_buffer[num_vec_size as usize][r];
                        for i in 0..remain {
                            buffer[i as usize] = buffer[i as usize];
                        }
                    }
                    load_store_res_buffer::<T, CONV_REGNUM, false>(
                        num_vec_size,
                        remain,
                        osw,
                        c * co_b + b * osb + l * osh + kp * (CONV_REGNUM as i64) * osw,
                        &mut res_buffer,
                        &mut out,
                    );
                }
            };

        let inp_cpy = inp.clone();
        let case1 = &case1;
        let case3 = move |b: i64, l: i64, c: i64, mut out: Pointer<T>| match wo_b_remain {
            1 => {
                case3_helper::<T, 1>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 1, false>,
                    micro_kernel_1_dyn::<T, 1>,
                    micro_kernel_1_with_buffer::<T>,
                    &mut out,
                );
            }
            2 => {
                case3_helper::<T, 2>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 2, false>,
                    micro_kernel_2_dyn::<T, 2>,
                    micro_kernel_2_with_buffer::<T>,
                    &mut out,
                );
            }
            3 => {
                case3_helper::<T, 3>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 3, false>,
                    micro_kernel_3_dyn::<T, 3>,
                    micro_kernel_3_with_buffer::<T>,
                    &mut out,
                );
            }
            4 => {
                case3_helper::<T, 4>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 4, false>,
                    micro_kernel_4_dyn::<T, 4>,
                    micro_kernel_4_with_buffer::<T>,
                    &mut out,
                );
            }
            5 => {
                case3_helper::<T, 5>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 2, false>,
                    micro_kernel_5_dyn::<T, 5>,
                    micro_kernel_5_with_buffer::<T>,
                    &mut out,
                );
            }
            6 => {
                case3_helper::<T, 6>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 6, false>,
                    micro_kernel_6_dyn::<T, 6>,
                    micro_kernel_6_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            7 => {
                case3_helper::<T, 7>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 7, false>,
                    micro_kernel_7_dyn::<T, 7>,
                    micro_kernel_7_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            8 => {
                case3_helper::<T, 8>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 8, false>,
                    micro_kernel_8_dyn::<T, 8>,
                    micro_kernel_8_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            9 => {
                case3_helper::<T, 9>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 9, false>,
                    micro_kernel_9_dyn::<T, 9>,
                    micro_kernel_9_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            10 => {
                case3_helper::<T, 10>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 10, false>,
                    micro_kernel_10_dyn::<T, 10>,
                    micro_kernel_10_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            11 => {
                case3_helper::<T, 11>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 11, false>,
                    micro_kernel_11_dyn::<T, 11>,
                    micro_kernel_11_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            12 => {
                case3_helper::<T, 12>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 12, false>,
                    micro_kernel_12_dyn::<T, 12>,
                    micro_kernel_12_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            13 => {
                case3_helper::<T, 13>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 13, false>,
                    micro_kernel_13_dyn::<T, 13>,
                    micro_kernel_13_with_buffer::<T>,
                    &mut out,
                );
            }
            #[cfg(any(target_feature = "avx512f", target_arch = "aarch64"))]
            14 => {
                case3_helper::<T, 14>(
                    [kernel_height, kernel_width],
                    [b, l, c],
                    [isb, ish, isw],
                    [osb, osh, osw],
                    [step_width, step_height],
                    [dh, dw],
                    co_b,
                    num_wo_b,
                    co_b_remain,
                    wo_b_remain,
                    &inp_cpy,
                    load_store_res_buffer::<T, 14, false>,
                    micro_kernel_14_dyn::<T, 14>,
                    micro_kernel_14_with_buffer::<T>,
                    &mut out,
                );
            }
            _ => unimplemented!(),
        };
        #[rustfmt::skip]
        (0..outer).into_par_iter().for_each(|idx| {
            let b = idx / (num_co_b * out_height);
            let l = (idx / num_co_b) % out_height;
            let c = idx % num_co_b;
            // println!("co_b_remain == 0: {}, wo_b_remain == 0: {}", co_b_remain == 0, wo_b_remain == 0);
            match (co_b_remain == 0, wo_b_remain == 0) {
                (true, true) => {
                    if co_b > 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>,  out.clone());
                    } else {
                        assert_eq!(co_b, 1);
                        case0_remain1(b, l, c,  out.clone());
                    }
                }
                (true, false) => {
                    if co_b > 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>,  out.clone());
                        case1(b, l, c, out.clone());
                    } else {
                        case0_remain1(b, l, c,  out.clone());
                        case1_remain1(b, l, c, out.clone());
                    }
                }
                (false, true) => {
                    // co_b_remain must be 0 if co_b is 1
                    assert!(co_b > 1);
                    if c < num_co_b - 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>, out.clone());
                    } else {
                        let remain = co_b_remain % (T::Vec::SIZE as i64);
                        if remain == 0 {
                            case0(b, l, c,  num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>, out.clone());
                            case0(b, l, num_co_b,  num_vec_size, micro_kernel_regnum::<T, CONV_REGNUM>, out.clone());
                        } else {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>, out.clone());
                            case2(b, l, num_co_b, num_vec_size, remain,  out.clone());
                        }
                    }
                }
                (false, false) => {
                    // co_b_remain must be 0 if co_b is 1
                    assert!(co_b > 1);
                    if c < num_co_b - 1 {
                        case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>,  out.clone());
                        case1(b, l, c,out.clone());
                    } else {
                        let remain = co_b_remain % (T::Vec::SIZE as i64);
                        if remain == 0 {
                            case0(b, l, c,  num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>, out.clone());
                            case1(b, l, c, out.clone());
                            case0(b, l, num_co_b,  num_vec_size, micro_kernel_regnum::<T, CONV_REGNUM>,  out.clone());
                            case3(b, l, num_co_b, out.clone());
                        } else {
                            case0(b, l, c, num_co_rb, micro_kernel_regnum::<T, CONV_REGNUM>,  out.clone());
                            case1(b, l, c,  out.clone());
                            case2(b, l, num_co_b, num_vec_size, remain, out.clone());
                            case3(b, l, num_co_b,out.clone());
                        }
                    }
                }
            }
        });

        Ok(output)
    }
}
