use std::sync::Arc;

use super::conv_config::Conv2dConfig;
use crate::tensor_base::_Tensor;
use duplicate::duplicate_item;
use rayon::prelude::*;
use tensor_common::err_handler::ErrHandler;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
use tensor_common::pointer::Pointer;
use tensor_common::shape_utils::mt_intervals;
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
    pub fn iconv2d(
        &self,
        kernels: &_Tensor<T>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        _: Option<&Conv2dConfig<T>>,
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
        let kernel = kernels.ptr();

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
        const OW_BLOCK: usize = 5;
        #[cfg(target_os = "macos")]
        const OC_NVEC: usize = 4;
        #[cfg(not(target_os = "macos"))]
        const OC_NVEC: usize = 2;
        const IC_NVEC: usize = 2;

        let l1_cache_size =
            cache_size::l1_cache_size().unwrap_or(64 * 1024) / core::mem::size_of::<T>();

        let inp_used = (OW_BLOCK as i64)
            * (IC_NVEC as i64)
            * (T::Vec::SIZE as i64)
            * kernel_height
            * kernel_width
            * OH_BLOCK;
        let kernel_used = (OW_BLOCK as i64)
            * (OC_NVEC as i64)
            * (T::Vec::SIZE as i64)
            * kernel_height
            * kernel_width;
        let out_used = (OW_BLOCK as i64) * (OC_NVEC as i64) * (T::Vec::SIZE as i64) * OH_BLOCK;
        let num_oc = (out_channels as usize)
            .div_ceil(OC_NVEC * T::Vec::SIZE)
            .max(1) as i64;
        let total = (kernel_used + out_used) * num_oc + inp_used;
        let optimal_num_oc = if total < (l1_cache_size as i64) {
            num_oc
        } else {
            (((l1_cache_size as i64) - inp_used) / (kernel_used + out_used)).max(1)
        };
        let num_opt_oc = num_oc / optimal_num_oc;
        let mut intervals = mt_intervals(num_oc as usize, num_opt_oc as usize); // we can use thread divide algo to get the intervals
        intervals.iter_mut().for_each(|(start, end)| {
            *start *= OC_NVEC * T::Vec::SIZE;
            *end *= OC_NVEC * T::Vec::SIZE;
            if *end >= (out_channels as usize) {
                *end = out_channels as usize;
            }
        });
        let intervals = Arc::new(intervals);
        let num_oh = (out_height as usize).div_ceil(OH_BLOCK as usize) as i64;
        let outer = batch * num_oh;
        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * OH_BLOCK;
            let l_end = (ll + OH_BLOCK).min(out_height);
            for ii in (0..in_channels).step_by(T::Vec::SIZE * IC_NVEC) {
                let i_end = (ii + (T::Vec::SIZE as i64) * (IC_NVEC as i64)).min(in_channels);
                for (jj_start, jj_end) in intervals.iter() {
                    let oc_remain = ((*jj_end - *jj_start) % (T::Vec::SIZE * OC_NVEC)) as i64;
                    let jj_start = *jj_start as i64;
                    let jj_end = *jj_end as i64;
                    let full_oc_end = jj_end - oc_remain;
                    for k in (0..out_width).step_by(OW_BLOCK) {
                        let k_end = (k + (OW_BLOCK as i64)).min(out_width);
                        let ow_remain = k_end - k;

                        // Main loop for full OC_NVEC blocks
                        match ow_remain {
                            5 => {
                                for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * OC_NVEC) {
                                    for l in ll..l_end {
                                        micro_kernel_5x2::<T, OC_NVEC>(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ks0, ks1, ks2],
                                            &mut out,
                                            &inp,
                                            &kernel,
                                        );
                                    }
                                }
                            }
                            4 => {
                                for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * OC_NVEC) {
                                    for l in ll..l_end {
                                        micro_kernel_4x2::<T, OC_NVEC>(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ks0, ks1, ks2],
                                            &mut out,
                                            &inp,
                                            &kernel,
                                        );
                                    }
                                }
                            }
                            3 => {
                                for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * OC_NVEC) {
                                    for l in ll..l_end {
                                        micro_kernel_3x2::<T, OC_NVEC>(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ks0, ks1, ks2],
                                            &mut out,
                                            &inp,
                                            &kernel,
                                        );
                                    }
                                }
                            }
                            2 => {
                                for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * OC_NVEC) {
                                    for l in ll..l_end {
                                        micro_kernel_2x2::<T, OC_NVEC>(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ks0, ks1, ks2],
                                            &mut out,
                                            &inp,
                                            &kernel,
                                        );
                                    }
                                }
                            }
                            1 => {
                                for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * OC_NVEC) {
                                    for l in ll..l_end {
                                        micro_kernel_1x4::<T, OC_NVEC>(
                                            [ii, i_end],
                                            [kernel_height, kernel_width],
                                            [b, l, k, j],
                                            [osb, osh, osw],
                                            [step_height, step_width],
                                            [isb, ish, isw],
                                            [ks0, ks1, ks2],
                                            &mut out,
                                            &inp,
                                            &kernel,
                                        );
                                    }
                                }
                            }
                            _ => unreachable!(),
                        }
                        // Handle remaining OC
                        if oc_remain > 0 {
                            let j_start = out_channels - oc_remain;
                            match ow_remain {
                                5 => {
                                    for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                        let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                        for l in ll..l_end {
                                            micro_kernel_5_scalar::<T>(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ks0, ks1, ks2],
                                                oc_end - j,
                                                &mut out,
                                                &inp,
                                                &kernel,
                                            );
                                        }
                                    }
                                }
                                4 => {
                                    for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                        let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                        for l in ll..l_end {
                                            micro_kernel_4_scalar::<T>(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ks0, ks1, ks2],
                                                oc_end - j,
                                                &mut out,
                                                &inp,
                                                &kernel,
                                            );
                                        }
                                    }
                                }
                                3 => {
                                    for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                        let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                        for l in ll..l_end {
                                            micro_kernel_3_scalar::<T>(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ks0, ks1, ks2],
                                                oc_end - j,
                                                &mut out,
                                                &inp,
                                                &kernel,
                                            );
                                        }
                                    }
                                }
                                2 => {
                                    for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                        let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                        for l in ll..l_end {
                                            micro_kernel_2_scalar::<T>(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ks0, ks1, ks2],
                                                oc_end - j,
                                                &mut out,
                                                &inp,
                                                &kernel,
                                            );
                                        }
                                    }
                                }
                                1 => {
                                    for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                        let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                        for l in ll..l_end {
                                            micro_kernel_1_scalar::<T>(
                                                [ii, i_end],
                                                [kernel_height, kernel_width],
                                                [b, l, k, j],
                                                [osb, osh, osw],
                                                [step_height, step_width],
                                                [isb, ish, isw],
                                                [ks0, ks1, ks2],
                                                oc_end - j,
                                                &mut out,
                                                &inp,
                                                &kernel,
                                            );
                                        }
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
        });
        Ok(output)
    }
}

macro_rules! repeat_inp {
    ($name:ident, $is3:expr, $step_width_m:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                T::Vec::splat($name[$is3 + $idx * $step_width_m]),
            )*)
        }
    };
}

macro_rules! repeat_kernel {
    ($name:ident, $kr3:expr, $vec_size:expr, [$($idx:expr),*]) => {
        paste::paste! {
            ($(
                T::Vec::splat($name[$kr3 + $idx * $vec_size]),
            )*)
        }
    };
}

macro_rules! repeat_results {
    ($results:ident, $inp:ident, $kernel:ident, [$vidx:literal, $($v:literal),*], [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $results[$vidx][$idx] = $inp.$idx.mul_add($kernel.$vidx, $results[$vidx][$idx]);
            )*
            repeat_results!($results, $inp, $kernel, [$($v),*], [$($idx),*]);
        }
    };
    ($results:ident, $inp:ident, $kernel:ident, [$vidx:literal], [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $results[$vidx][$idx] = $inp.$idx.mul_add($kernel.$vidx, $results[$vidx][$idx]);
            )*
        }
    };
}

macro_rules! repeat_results_scalar {
    ($results:ident, $inp:ident, $kernel:ident, [$($idx:expr),*]) => {
        paste::paste! {
            $(
                $results[$idx] = $inp.$idx.mul_add($kernel, $results[$idx]);
            )*
        }
    };
}

#[
    duplicate_item(
        template_function   ow_block    inp_place_holder      kernel_place_holder              oc;
        [micro_kernel_5x2]    [5]      [[0, 1, 2, 3, 4]]       [[0, 1]]                     [[0, 1]];
        [micro_kernel_4x2]    [4]      [[0, 1, 2, 3]]          [[0, 1]]                     [[0, 1]];
        [micro_kernel_3x2]    [3]      [[0, 1, 2]]             [[0, 1]]                     [[0, 1]];
        [micro_kernel_2x2]    [2]      [[0, 1]]                [[0, 1]]                     [[0, 1]];
        [micro_kernel_1x2]    [1]      [[0]]                   [[0, 1]]                     [[0, 1]];
        [micro_kernel_5x4]    [5]      [[0, 1, 2, 3, 4]]       [[0, 1, 2, 3]]               [[0, 1, 2, 3]];
        [micro_kernel_4x4]    [4]      [[0, 1, 2, 3]]          [[0, 1, 2, 3]]               [[0, 1, 2, 3]];
        [micro_kernel_3x4]    [3]      [[0, 1, 2]]             [[0, 1, 2, 3]]               [[0, 1, 2, 3]];
        [micro_kernel_2x4]    [2]      [[0, 1]]                [[0, 1, 2, 3]]               [[0, 1, 2, 3]];
        [micro_kernel_1x4]    [1]      [[0]]                   [[0, 1, 2, 3]]               [[0, 1, 2, 3]];
        [micro_kernel_5x8]    [5]      [[0, 1, 2, 3, 4]]       [[0, 1, 2, 3, 4, 5, 6, 7]]   [[0, 1, 2, 3, 4, 5, 6, 7]];
        [micro_kernel_4x8]    [4]      [[0, 1, 2, 3]]          [[0, 1, 2, 3, 4, 5, 6, 7]]   [[0, 1, 2, 3, 4, 5, 6, 7]];
        [micro_kernel_3x8]    [3]      [[0, 1, 2]]             [[0, 1, 2, 3, 4, 5, 6, 7]]   [[0, 1, 2, 3, 4, 5, 6, 7]];
        [micro_kernel_2x8]    [2]      [[0, 1]]                [[0, 1, 2, 3, 4, 5, 6, 7]]   [[0, 1, 2, 3, 4, 5, 6, 7]];
        [micro_kernel_1x8]    [1]      [[0]]                   [[0, 1, 2, 3, 4, 5, 6, 7]]   [[0, 1, 2, 3, 4, 5, 6, 7]];
    )
]
#[inline]
fn template_function<T: CommonBounds, const OC_NVEC: usize>(
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [ks0, ks1, ks2]: [i64; 3],
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &Pointer<T>,
) {
    const OW_BLOCK: usize = ow_block;
    let mut results = if ii == 0 {
        [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_NVEC]
    } else {
        let mut ret = [[T::Vec::splat(T::ZERO); OW_BLOCK]; OC_NVEC];
        for v in 0..OC_NVEC {
            for kk in 0..OW_BLOCK as i64 {
                ret[v as usize][kk as usize] = unsafe {
                    T::Vec::from_ptr(
                        &out[b * osb
                            + l * osh
                            + (k + kk) * osw
                            + j
                            + v as i64 * T::Vec::SIZE as i64] as *const _
                            as *const T,
                    )
                }; // prettier-ignore
            }
        }
        ret
    };
    let is0 = b * isb + l * step_height * ish + k * step_width * isw;
    let kr0 = j;
    for n in 0..kh {
        let is1 = is0 + n * ish;
        let kr1 = n * ks0 + kr0;
        for m in 0..kw {
            let is2 = is1 + m * isw;
            let kr2 = kr1 + m * ks1;
            for i in ii..i_end {
                let is3 = is2 + i;
                let kr3 = i * ks2 + kr2;
                let inp = repeat_inp!(inp, is3, step_width * isw, inp_place_holder);
                let kernel = repeat_kernel!(kernel, kr3, T::Vec::SIZE as i64, kernel_place_holder);
                repeat_results!(results, inp, kernel, oc, inp_place_holder);
            }
        }
    }
    for v in 0..OC_NVEC as i64 {
        for kk in 0..OW_BLOCK as i64 {
            let out_vec = &mut out[b * osb + l * osh + (k + kk) * osw + j + v * T::Vec::SIZE as i64]
                as *mut _ as *mut T::Vec; // prettier-ignore
            unsafe {
                out_vec.write_unaligned(results[v as usize][kk as usize]);
            }
        }
    }
}

#[
    duplicate_item(
        template_function          inp_place_holder      kernel_place_holder   oc;
        [micro_kernel_5_scalar]    [[0, 1, 2, 3, 4]]       [[0, 1]]        [[0]];
        [micro_kernel_4_scalar]    [[0, 1, 2, 3]]          [[0, 1]]        [[0]];
        [micro_kernel_3_scalar]    [[0, 1, 2]]             [[0, 1]]        [[0]];
        [micro_kernel_2_scalar]    [[0, 1]]                [[0, 1]]        [[0]];
        [micro_kernel_1_scalar]    [[0]]                   [[0, 1]]        [[0]];
    )
]
#[inline]
fn template_function<T: CommonBounds>(
    [ii, i_end]: [i64; 2],
    [kh, kw]: [i64; 2],
    [b, l, k, j]: [i64; 4],
    [osb, osh, osw]: [i64; 3],
    [step_height, step_width]: [i64; 2],
    [isb, ish, isw]: [i64; 3],
    [ks0, ks1, ks2]: [i64; 3],
    oc_end: i64,
    out: &mut Pointer<T>,
    inp: &Pointer<T>,
    kernel: &Pointer<T>,
) {
    const OW_BLOCK: usize = 5;
    let mut results = if ii == 0 {
        [T::Vec::splat(T::ZERO); OW_BLOCK]
    } else {
        let mut ret = [T::Vec::splat(T::ZERO); OW_BLOCK];
        for kk in 0..OW_BLOCK as i64 {
            for v in 0..oc_end {
                ret[kk as usize][v as usize] = out[b * osb + l * osh + (k + kk) * osw + j];
            }
        }
        ret
    };
    let is0 = b * isb + l * step_height * ish + k * step_width * isw;
    let kr0 = j;
    for n in 0..kh {
        let is1 = is0 + n * ish;
        let kr1 = n * ks0 + kr0;
        for m in 0..kw {
            let is2 = is1 + m * isw;
            let kr2 = kr1 + m * ks1;
            for i in ii..i_end {
                let is3 = is2 + i;
                let kr3 = i * ks2 + kr2;
                let inp = repeat_inp!(inp, is3, step_width * isw, inp_place_holder);
                let mut kernel0 = T::Vec::splat(T::ZERO);
                for v in 0..oc_end {
                    kernel0[v as usize] = kernel[n * ks0 + m * ks1 + i * ks2 + j];
                }
                repeat_results_scalar!(results, inp, kernel0, inp_place_holder);
            }
        }
    }
    for kk in 0..OW_BLOCK as i64 {
        for v in 0..oc_end {
            out[b * osb + l * osh + (k + kk) * osw + j] = results[kk as usize][v as usize];
        }
    }
}