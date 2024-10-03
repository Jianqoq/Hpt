use std::sync::Arc;

use super::conv_config::Conv2dConfig;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_full_oc_kernel_dispatch;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_remain_oc_kernel_dispatch;
use crate::tensor_base::_Tensor;
use crate::REGNUM;
use crate::SIMD_WIDTH;
use rayon::prelude::*;
use tensor_common::err_handler::ErrHandler;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
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

        let cache_line_size = cache_size::l1_cache_line_size().unwrap_or(64);
        let (ow_block, oc_nvec) = optimize_ow_block_and_oc_nvec(
            cache_line_size,
            SIMD_WIDTH,
            REGNUM
        );
        let ic_nvec = oc_nvec;

        let full_oc_fn = iconv2d_full_oc_kernel_dispatch(oc_nvec, ow_block);
        let full_oc_remain_fn = iconv2d_full_oc_kernel_dispatch(
            oc_nvec,
            (out_width as usize) % ow_block
        );
        let partial_oc_fn = iconv2d_remain_oc_kernel_dispatch(ow_block);
        let partial_oc_remain_fn = iconv2d_remain_oc_kernel_dispatch(
            (out_width as usize) % ow_block
        );

        let l1_cache_size =
            cache_size::l1_cache_size().unwrap_or(10 * 1024 /* 10kb */) / core::mem::size_of::<T>();

        let inp_used =
            (ow_block as i64) *
            (ic_nvec as i64) *
            (T::Vec::SIZE as i64) *
            kernel_height *
            kernel_width *
            OH_BLOCK;
        let kernel_used =
            (ow_block as i64) *
            (oc_nvec as i64) *
            (T::Vec::SIZE as i64) *
            kernel_height *
            kernel_width;
        let out_used = (ow_block as i64) * (oc_nvec as i64) * (T::Vec::SIZE as i64) * OH_BLOCK;
        let num_oc = (out_channels as usize).div_ceil(oc_nvec * T::Vec::SIZE).max(1) as i64;
        let total = (kernel_used + out_used) * num_oc + inp_used;
        // println!("total: {}", total);
        let optimal_num_oc = if total < (l1_cache_size as i64) {
            num_oc
        } else {
            (((l1_cache_size as i64) - inp_used) / (kernel_used + out_used)).max(1)
        };
        let num_opt_oc = num_oc / optimal_num_oc;
        let mut intervals = mt_intervals(num_oc as usize, num_opt_oc as usize); // we can use thread divide algo to get the intervals
        intervals.iter_mut().for_each(|(start, end)| {
            *start *= oc_nvec * T::Vec::SIZE;
            *end *= oc_nvec * T::Vec::SIZE;
            if *end >= (out_channels as usize) {
                *end = out_channels as usize;
            }
        });
        let intervals = Arc::new(intervals);
        let num_oh = (out_height as usize).div_ceil(OH_BLOCK as usize) as i64;
        let outer = batch * num_oh;
        let full_ow_blocks_end = out_width - (out_width % (ow_block as i64));
        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * OH_BLOCK;
            let l_end = (ll + OH_BLOCK).min(out_height);
            for ii in (0..in_channels).step_by(T::Vec::SIZE * ic_nvec) {
                let i_end = (ii + (T::Vec::SIZE as i64) * (ic_nvec as i64)).min(in_channels);
                for (jj_start, jj_end) in intervals.iter() {
                    let oc_remain = ((*jj_end - *jj_start) % (T::Vec::SIZE * oc_nvec)) as i64;
                    let jj_start = *jj_start as i64;
                    let jj_end = *jj_end as i64;
                    let full_oc_end = jj_end - oc_remain;
                    for k in (0..full_ow_blocks_end).step_by(ow_block) {
                        // Main loop for full oc_nvec blocks
                        for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * oc_nvec) {
                            for l in ll..l_end {
                                full_oc_fn(
                                    [ii, i_end],
                                    [kernel_height, kernel_width],
                                    [b, l, k, j],
                                    [osb, osh, osw],
                                    [step_height, step_width],
                                    [isb, ish, isw],
                                    [ks0, ks1, ks2],
                                    &mut out,
                                    &inp,
                                    &kernel
                                );
                            }
                        }
                        // Handle remaining OC
                        if oc_remain > 0 {
                            let j_start = out_channels - oc_remain;
                            for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                for l in ll..l_end {
                                    partial_oc_fn(
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
                                        &kernel
                                    );
                                }
                            }
                        }
                    }
                    for k in (full_ow_blocks_end..out_width).step_by(ow_block) {
                        // Main loop for full oc_nvec blocks
                        for j in (jj_start..full_oc_end).step_by(T::Vec::SIZE * oc_nvec) {
                            for l in ll..l_end {
                                full_oc_remain_fn(
                                    [ii, i_end],
                                    [kernel_height, kernel_width],
                                    [b, l, k, j],
                                    [osb, osh, osw],
                                    [step_height, step_width],
                                    [isb, ish, isw],
                                    [ks0, ks1, ks2],
                                    &mut out,
                                    &inp,
                                    &kernel
                                );
                            }
                        }
                        // Handle remaining OC
                        if oc_remain > 0 {
                            let j_start = out_channels - oc_remain;
                            for j in (j_start..out_channels).step_by(T::Vec::SIZE) {
                                let oc_end = (j + (T::Vec::SIZE as i64)).min(out_channels);
                                for l in ll..l_end {
                                    partial_oc_remain_fn(
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
                                        &kernel
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
        Ok(output)
    }
}

fn optimize_ow_block_and_oc_nvec(
    cache_line_size: usize,
    simd_width: usize,
    reg_num: usize
) -> (usize, usize) {
    let initial_oc_nvec = cache_line_size / (simd_width / 8);
    let mut best_ow_block = 4;
    let mut best_oc_nvec = 1;
    let mut best_utilization = 0;

    for oc_nvec in 1..=initial_oc_nvec {
        let max_ow_block = ((reg_num - oc_nvec) / (oc_nvec + 1)).max(4);
        for ow_block in 4..=max_ow_block {
            let total_regs_used = (ow_block + 1) * oc_nvec + ow_block;
            if total_regs_used <= reg_num {
                let utilization = ow_block * oc_nvec;
                if utilization > best_utilization {
                    best_utilization = utilization;
                    best_ow_block = ow_block;
                    best_oc_nvec = oc_nvec;
                }
            }
        }
    }

    (best_ow_block, best_oc_nvec)
}
