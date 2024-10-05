use super::conv_config::Conv2dConfig;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_full_oc_kernel_dispatch;
use crate::ops::cpu::kernels::iconv_kernels::iconv2d_remain_oc_kernel_dispatch;
use crate::tensor_base::_Tensor;
use rayon::prelude::*;
use tensor_common::err_handler::ErrHandler;
use tensor_common::err_handler::ErrHandler::InvalidInputShape;
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

        let ic_nvec = 16;
        let mut oc_nvec = 2;
        let mut ow_block = 5;

        // let inp_used =
        //     (ow_block as i64) *
        //     (ic_nvec as i64) *
        //     (T::Vec::SIZE as i64) *
        //     kernel_height *
        //     kernel_width *
        //     OH_BLOCK;
        // let kernel_used =
        //     (ow_block as i64) *
        //     (oc_nvec as i64) *
        //     (T::Vec::SIZE as i64) *
        //     kernel_height *
        //     kernel_width;
        // let out_used = (ow_block as i64) * (oc_nvec as i64) * (T::Vec::SIZE as i64) * OH_BLOCK;
        // println!("inp_used: {}, kernel_used: {}, out_used: {}", inp_used, kernel_used, out_used);

        let full_oc_kernel = iconv2d_full_oc_kernel_dispatch(&mut oc_nvec, &mut ow_block, false)
            .expect(&format!(
                "unable to find iconv2d_microkernel_{}x{}",
                ow_block, oc_nvec
            ));
        let full_oc_kernel_ow_remain = iconv2d_full_oc_kernel_dispatch(
            &mut oc_nvec,
            &mut ((out_width as usize) % ow_block),
            true,
        );
        if full_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let partial_oc_kernel = iconv2d_remain_oc_kernel_dispatch(&mut ow_block);
        let partial_oc_kernel_ow_remain =
            iconv2d_remain_oc_kernel_dispatch(&mut ((out_width as usize) % ow_block));
        if partial_oc_kernel_ow_remain.is_none() {
            assert_eq!((out_width as usize) % ow_block, 0);
        }
        let num_oh = (out_height + OH_BLOCK - 1) / OH_BLOCK;
        let outer = batch * num_oh;
        let out_width_full_end = out_width - (out_width % (ow_block as i64));
        let oc_remain = out_channels % ((T::Vec::SIZE as i64) * (oc_nvec as i64));

        (0..outer).into_par_iter().for_each(|idx| {
            let mut out = out.clone();
            let b = idx / num_oh;
            let ll = idx % num_oh;
            let ll = ll * OH_BLOCK;
            let l_end = (ll + OH_BLOCK).min(out_height);
            for ii in (0..in_channels).step_by(T::Vec::SIZE * ic_nvec) {
                let i_end = (ii + (T::Vec::SIZE as i64) * (ic_nvec as i64)).min(in_channels);
                for k in (0..out_width_full_end).step_by(ow_block) {
                    for j in (0..out_channels - oc_remain).step_by(T::Vec::SIZE * oc_nvec) {
                        for l in ll..l_end {
                            full_oc_kernel(
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
                    if let Some(partial_oc_kernel) = partial_oc_kernel {
                        for j in
                            (out_channels - oc_remain..out_channels).step_by(T::Vec::SIZE * oc_nvec)
                        {
                            for l in ll..l_end {
                                partial_oc_kernel(
                                    [ii, i_end],
                                    [kernel_height, kernel_width],
                                    [b, l, k, j],
                                    [osb, osh, osw],
                                    [step_height, step_width],
                                    [isb, ish, isw],
                                    [ks0, ks1, ks2],
                                    oc_remain,
                                    &mut out,
                                    &inp,
                                    &kernel,
                                );
                            }
                        }
                    }
                }
                if let Some(full_oc_kernel_ow_remain) = full_oc_kernel_ow_remain {
                    for k in (out_width_full_end..out_width).step_by(ow_block) {
                        for j in (0..out_channels - oc_remain).step_by(T::Vec::SIZE * oc_nvec) {
                            for l in ll..l_end {
                                full_oc_kernel_ow_remain(
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
                        if let Some(partial_oc_kernel_ow_remain) = partial_oc_kernel_ow_remain {
                            for j in (out_channels - oc_remain..out_channels)
                                .step_by(T::Vec::SIZE * oc_nvec)
                            {
                                for l in ll..l_end {
                                    partial_oc_kernel_ow_remain(
                                        [ii, i_end],
                                        [kernel_height, kernel_width],
                                        [b, l, k, j],
                                        [osb, osh, osw],
                                        [step_height, step_width],
                                        [isb, ish, isw],
                                        [ks0, ks1, ks2],
                                        oc_remain,
                                        &mut out,
                                        &inp,
                                        &kernel,
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

fn out_used(lb: usize, jb: usize, oc_nvec: usize, owb: usize, line_size: usize) -> usize {
    lb * jb * oc_nvec * owb * line_size
}

fn inp_used(
    lb: usize,
    owb: usize,
    ic_nvec: usize,
    kh: usize,
    kw: usize,
    line_size: usize,
) -> usize {
    owb * ic_nvec * kh * kw * line_size * lb
}

fn kernel_used(oc_nvec: usize, ic_nvec: usize, kh: usize, kw: usize, line_size: usize) -> usize {
    oc_nvec * ic_nvec * kh * kw * line_size
}
