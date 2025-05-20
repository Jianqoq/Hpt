use hpt_common::error::base::TensorError;
use hpt_common::error::shape::ShapeError;
use hpt_traits::tensor::CommonBounds;
use hpt_traits::tensor::TensorInfo;
use hpt_types::dtype::DType;
use hpt_types::dtype::ToDType;
use hpt_types::into_scalar::Cast;
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::type_promote::FloatOutUnary;
use hpt_types::type_promote::NormalOutPromote;

use crate::Tensor;
use crate::ops::tensor::conv2d::conv2d_mp;
use crate::ops::tensor::matmul::microkernel_trait::MatmulMicroKernel;

use super::batchnorm_conv2d::batchnorm_conv2d;
use super::conv2d_group::conv2d_group;
use super::microkernel_trait::Conv2dMicroKernel;
use super::utils::cal_conv2d_output_shape;
use super::{conv2d_direct, conv2d_img2col};

pub(crate) fn conv2d<T: CommonBounds + Conv2dMicroKernel + ToDType>(
    input: &Tensor,
    kernels: &Tensor,
    bias: Option<&Tensor>,
    steps: [i64; 2],
    padding: [(i64, i64); 2],
    dilation: [i64; 2],
    post_scalar: Option<fn(T) -> T>,
    post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
) -> Result<Tensor, TensorError>
where
    T: MatmulMicroKernel,
{
    let input = if !input.is_contiguous() || input.parent.is_some() {
        input.contiguous()?
    } else {
        input.clone()
    };
    if bias.is_some() {
        ShapeError::check_contiguous(
            "Conv2d requires bias tensor to be contiguous. ".to_string(),
            bias.unwrap().layout(),
        )?;
    }
    let img_shape = input.shape();
    ShapeError::check_dim(4, img_shape.len())?;
    let batch = img_shape[0];
    let img_height = img_shape[1];
    let img_width = img_shape[2];
    let img_channels = img_shape[3];
    let kernel_shape = kernels.shape();
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let in_channels = kernel_shape[2];
    let out_channels = kernel_shape[3];
    if in_channels != img_channels {
        return Err((ShapeError::ConvError {
            message: format!(
                "kernel in_channel {} not match input in_channel {}",
                in_channels, img_channels
            ),
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let ((ph_start, ph_end), (pw_start, pw_end)) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);

    let (out_height, out_width) = cal_conv2d_output_shape(
        img_height,
        img_width,
        kh,
        kw,
        &[(ph_start, ph_end), (pw_start, pw_end)],
        &[step_height, step_width],
        &[dh, dw],
    );
    if out_height <= 0 || out_width <= 0 {
        return Err((ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        })
        .into());
    }
    let output = Tensor::empty(
        &[batch, out_height, out_width, out_channels],
        input.dtype,
        input.device.clone(),
    )?;
    let img2col_buffer_size = kh * kw * in_channels * out_height * out_width;
    let direct_buffer_size = kh * kw * in_channels * out_channels;
    if img2col_buffer_size < direct_buffer_size {
        let ret = conv2d_img2col::conv2d(
            &input,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            batch,
            img_height,
            img_width,
            img_channels,
            out_channels,
            kh,
            kw,
            post_scalar,
            post_vec,
            output,
        );
        ret
    } else {
        let ret = conv2d_direct::conv2d(
            &input,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            batch,
            img_height,
            img_width,
            img_channels,
            out_channels,
            kh,
            kw,
            post_scalar,
            post_vec,
            output,
        );
        ret
    }
}

impl Tensor {
    pub fn conv2d(
        &self,
        kernels: &Tensor,
        bias: Option<&Tensor>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "i8")]
            DType::I8 => {
                self.conv2d_post::<i8>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "u8")]
            DType::U8 => {
                self.conv2d_post::<u8>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "f32")]
            DType::F32 => {
                self.conv2d_post::<f32>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                self.conv2d_post::<half::f16>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "bf16")]
            DType::BF16 => {
                self.conv2d_post::<half::bf16>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "bool")]
            DType::Bool => {
                self.conv2d_post::<bool>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "i16")]
            DType::I16 => {
                self.conv2d_post::<i16>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "u16")]
            DType::U16 => {
                self.conv2d_post::<u16>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "i32")]
            DType::I32 => {
                self.conv2d_post::<i32>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "u32")]
            DType::U32 => {
                self.conv2d_post::<u32>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "i64")]
            DType::I64 => {
                self.conv2d_post::<i64>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "u64")]
            DType::U64 => {
                self.conv2d_post::<u64>(kernels, bias, steps, padding, dilation, None, None)
            }
            #[cfg(feature = "f64")]
            DType::F64 => {
                self.conv2d_post::<f64>(kernels, bias, steps, padding, dilation, None, None)
            }
            _ => panic!("unsupported dtype: {:?}", self.dtype),
        }
    }

    pub fn conv2d_post<T: CommonBounds + ToDType + Conv2dMicroKernel + MatmulMicroKernel>(
        &self,
        kernels: &Tensor,
        bias: Option<&Tensor>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        post_scalar: Option<fn(T) -> T>,
        post_vec: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Tensor, TensorError> {
        let t_dtype = self.dtype;
        assert_eq!(self.dtype, t_dtype);
        assert_eq!(kernels.dtype, t_dtype);
        if let Some(bias) = bias {
            assert_eq!(bias.dtype, t_dtype);
        }
        if t_dtype == DType::F16
            && (cfg!(target_feature = "neon") && cfg!(target_feature = "fp16"))
        {
            conv2d_mp::conv2d_mp::<T>(
                self,
                kernels,
                bias,
                steps,
                padding,
                dilation,
                post_scalar,
                post_vec,
            )
        } else if t_dtype == DType::BF16 {
            conv2d_mp::conv2d_mp::<T>(
                self,
                kernels,
                bias,
                steps,
                padding,
                dilation,
                post_scalar,
                post_vec,
            )
        } else {
            conv2d::<T>(
                self,
                kernels,
                bias,
                steps,
                padding,
                dilation,
                post_scalar,
                post_vec,
            )
        }
    }

    pub fn conv2d_bn<T>(
        &self,
        kernels: &Self,
        mean: &Self,
        var: &Self,
        gamma: &Self,
        beta: &Self,
        bias: Option<&Self>,
        eps: T,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
    ) -> Result<Tensor, TensorError>
    where
        T: CommonBounds
            + Conv2dMicroKernel
            + MatmulMicroKernel
            + Cast<<T as NormalOutPromote>::Intermediate>
            + ToDType,
        <T as NormalOutPromote>::Intermediate: CommonBounds + Cast<T>,
        T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
        T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    {
        let t_dtype = T::to_dtype();
        assert_eq!(self.dtype, t_dtype);
        assert_eq!(kernels.dtype, t_dtype);
        assert_eq!(mean.dtype, t_dtype);
        assert_eq!(var.dtype, t_dtype);
        assert_eq!(gamma.dtype, t_dtype);
        assert_eq!(beta.dtype, t_dtype);
        if let Some(bias) = bias {
            assert_eq!(bias.dtype, t_dtype);
        }
        batchnorm_conv2d::<T>(
            self, kernels, mean, var, gamma, beta, bias, eps, steps, padding, dilation, None, None,
        )
    }

    pub fn conv2d_group(
        &self,
        kernels: &Self,
        bias: Option<&Self>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
    ) -> Result<Tensor, TensorError> {
        match self.dtype {
            #[cfg(feature = "bool")]
            DType::Bool => conv2d_group::<bool>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "i8")]
            DType::I8 => conv2d_group::<i8>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "u8")]
            DType::U8 => conv2d_group::<u8>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "i16")]
            DType::I16 => conv2d_group::<i16>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "u16")]
            DType::U16 => conv2d_group::<u16>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "i32")]
            DType::I32 => conv2d_group::<i32>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "u32")]
            DType::U32 => conv2d_group::<u32>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "i64")]
            DType::I64 => conv2d_group::<i64>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "u64")]
            DType::U64 => conv2d_group::<u64>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "f32")]
            DType::F32 => conv2d_group::<f32>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "f16")]
            DType::F16 => conv2d_group::<half::f16>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "bf16")]
            DType::BF16 => conv2d_group::<half::bf16>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            #[cfg(feature = "f64")]
            DType::F64 => conv2d_group::<f64>(
                self, kernels, bias, steps, padding, dilation, groups, None, None,
            ),
            _ => panic!("unsupported dtype: {:?}", self.dtype),
        }
    }
}
