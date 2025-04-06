#![allow(unused)]

use crate::backends::cuda::cuda_utils::{compute_kernel_launch_config, load_ptx_and_get_data};
use crate::{backends::common::conv::cal_conv2d_output_shape, tensor_base::_Tensor, CUDNN};
use cudarc::{
    cudnn::{ConvForward, Cudnn, CudnnDataType},
    driver::{CudaSlice, DeviceRepr, LaunchAsync},
};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};
use hpt_common::{error::shape::ShapeError, utils::conv_algos::ConvAlgo};
use hpt_cudakernels::CONV2D_BATCHNORM;
use hpt_traits::ops::{
    conv::CudaConvBatchNorm, creation::TensorCreator, shape_manipulate::ShapeManipulate,
    unary::Contiguous,
};
use hpt_traits::{
    ops::conv::{ConvBatchNorm, CudaConv},
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::{
    cuda_types::scalar::Scalar,
    dtype::CudaType,
    into_scalar::Cast,
    type_promote::{FloatOutBinary, FloatOutUnary, NormalOut},
};

pub(crate) fn conv2d<T, const DEVICE: usize, Al>(
    input: &_Tensor<T, Cuda, DEVICE, Al>,
    kernels: &_Tensor<T, Cuda, DEVICE, Al>,
    bias: Option<&_Tensor<T, Cuda, DEVICE, Al>>,
    steps: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    group_count: i64,
    math_type: cudarc::cudnn::sys::cudnnMathType_t,
    algo: Option<ConvAlgo>,
) -> Result<_Tensor<T, Cuda, DEVICE, Al>, hpt_common::error::base::TensorError>
where
    T: CommonBounds + DeviceRepr + CudaType + CudnnDataType,
    bool: Cast<T>,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
    Scalar<T>: NormalOut<Scalar<T>, Output = Scalar<T>>,
{
    if T::STR == "i64" || T::STR == "i32" || T::STR == "u8" || T::STR == "i8" || T::STR == "bool" {
        return Err(ShapeError::ConvError {
            message: format!("{} is not supported", T::STR),
            location: core::panic::Location::caller(),
        }
        .into());
    }

    ShapeError::check_contiguous(
        "Conv2d requires kernel tensor to be contiguous. ".to_string(),
        kernels.layout(),
    )?;
    let res_device = input.device();
    let cudnn = Cudnn::new(res_device).unwrap(); // when drop in test env, it throws error STATUS_STACK_BUFFER_OVERRUN, for now, I have to use this new
    let inp_shape = [
        input.shape()[0] as i32,
        input.shape()[1] as i32,
        input.shape()[2] as i32,
        input.shape()[3] as i32,
    ];
    let kernel_shape = [
        kernels.shape()[0] as i32,
        kernels.shape()[1] as i32,
        kernels.shape()[2] as i32,
        kernels.shape()[3] as i32,
    ];
    if kernel_shape[3] != inp_shape[3] && group_count == 1 {
        return Err(ShapeError::ConvError {
            message: format!(
                "kernel in_channel {} not match input in_channel {}",
                kernel_shape[3], inp_shape[3]
            ),
            location: core::panic::Location::caller(),
        }
        .into());
    }
    if inp_shape[3] % group_count as i32 != 0 {
        return Err(ShapeError::ConvError {
            message: format!(
                "Input channels ({}) must be divisible by groups ({})",
                inp_shape[3], group_count
            ),
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let (step_width, step_height) = (steps[0], steps[1]);
    let (ph, pw) = (padding[0], padding[1]);
    let (dh, dw) = (dilation[0], dilation[1]);
    let (out_height, out_width) = cal_conv2d_output_shape(
        inp_shape[1] as i64,
        inp_shape[2] as i64,
        kernel_shape[1] as i64,
        kernel_shape[2] as i64,
        &[(ph, ph), (pw, pw)],
        &[step_height, step_width],
        &[dh, dw],
    );
    if out_height <= 0 || out_width <= 0 {
        return Err(ShapeError::ConvError {
            message: if out_height <= 0 {
                "output height <= 0".to_string()
            } else {
                "output width <= 0".to_string()
            },
            location: core::panic::Location::caller(),
        }
        .into());
    }
    let output_shape = [
        inp_shape[0] as i32,
        out_height as i32,
        out_width as i32,
        kernel_shape[0] as i32,
    ];
    if let Some(bias) = bias {
        if bias.shape().len() != 1 {
            return Err(ShapeError::ConvError {
                message: "bias must be 1D tensor".to_string(),
                location: core::panic::Location::caller(),
            }
            .into());
        }
        if bias.shape()[0] as i32 != output_shape[3] {
            return Err(ShapeError::ConvError {
                message: format!(
                    "bias size {} must equal to output channel {}",
                    bias.shape()[0],
                    output_shape[3]
                ),
                location: core::panic::Location::caller(),
            }
            .into());
        }
    }

    let mut conv2d = cudnn
        .create_conv2d::<T>(
            [padding[0] as i32, padding[1] as i32],
            [steps[0] as i32, steps[1] as i32],
            [dilation[0] as i32, dilation[1] as i32],
            cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .expect("create conv2d failed");
    conv2d.set_group_count(group_count as i32);
    conv2d.set_math_type(math_type);
    let x = if input.is_contiguous() && input.parent().is_none() {
        cudnn
            .create_4d_tensor::<T>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]],
            )
            .expect("create tensor failed")
    } else {
        let strides = input.strides();
        cudnn
            .create_4d_tensor_ex::<T>(
                inp_shape,
                [
                    strides[0] as i32,
                    strides[1] as i32,
                    strides[2] as i32,
                    strides[3] as i32,
                ],
            )
            .expect("create tensor failed")
    };
    let kernel = cudnn
        .create_4d_filter::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
            [
                kernel_shape[0],
                kernel_shape[3],
                kernel_shape[1],
                kernel_shape[2],
            ],
        )
        .expect("create tensor failed");
    let output = cudnn
        .create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
            [
                output_shape[0],
                output_shape[3],
                output_shape[1],
                output_shape[2],
            ],
        )
        .expect("create tensor failed");
    let conv2d = ConvForward {
        conv: &conv2d,
        x: &x,
        w: &kernel,
        y: &output,
    };
    let algo = match algo {
        Some(algo) => match algo {
            ConvAlgo::ImplicitGemm => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            ConvAlgo::ImplicitPrecompGemm => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            ConvAlgo::Gemm => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            ConvAlgo::Direct => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            ConvAlgo::Fft => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            ConvAlgo::FftTiling => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            ConvAlgo::Winograd => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            ConvAlgo::WinogradNonFused => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            ConvAlgo::Count => cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
        },
        None => {
            match conv2d.pick_algorithm() {
                Ok(algo) => algo,
                Err(err) => {
                    panic!("pick algorithm failed: {:#?}", err);
                },
            }
        },
    };

    let workspace_size = conv2d
        .get_workspace_size(algo)
        .expect("get workspace size failed");
    let mut workspace = input.device().alloc_zeros::<u8>(workspace_size)?;

    let res = _Tensor::<T, Cuda, DEVICE, Al>::empty(&output_shape)?;
    let mut res_slice = unsafe {
        res.device()
            .upgrade_device_ptr::<T>(res.ptr().ptr as u64, res.size())
    };
    let inp_slice = unsafe {
        input
            .device()
            .upgrade_device_ptr::<T>(input.ptr().ptr as u64, input.size())
    };
    let kernel_slice = unsafe {
        kernels
            .device()
            .upgrade_device_ptr::<T>(kernels.ptr().ptr as u64, kernels.size())
    };
    unsafe {
        conv2d
            .launch::<CudaSlice<u8>, _, _, _>(
                algo,
                Some(&mut workspace),
                (T::ONE, T::ZERO),
                &inp_slice,
                &kernel_slice,
                &mut res_slice,
            )
            .expect("launch conv2d failed");
    }
    res_slice.leak();
    kernel_slice.leak();
    inp_slice.leak();
    if let Some(bias) = bias {
        return Ok(bias + res);
    }
    Ok(res)
}

impl<T, const DEVICE: usize, Al> CudaConv<T> for _Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + DeviceRepr + CudaType + CudnnDataType,
    bool: Cast<T>,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
    Scalar<T>: NormalOut<Scalar<T>, Output = Scalar<T>>,
{
    type Output = _Tensor<T, Cuda, DEVICE, Al>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        algo: Option<ConvAlgo>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d(
            self,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            1,
            cudarc::cudnn::sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
            algo,
        )
    }

    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
        algo: Option<ConvAlgo>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d(
            self,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            groups,
            cudarc::cudnn::sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
            algo,
        )
    }

    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        algo: Option<ConvAlgo>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        conv2d(
            self,
            kernels,
            bias,
            steps,
            padding,
            dilation,
            self.shape()[3],
            cudarc::cudnn::sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
            algo,
        )
    }

    fn conv2d_transpose(
        &self,
        kernels: &Self::Output,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        unimplemented!()
    }
}

impl<T, const DEVICE: usize, A> CudaConvBatchNorm<T> for _Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds + DeviceRepr + CudaType + CudnnDataType,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    Scalar<T>: NormalOut<Scalar<T>, Output = Scalar<T>>,
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = _Tensor<T, Cuda, DEVICE, A>;
    fn batchnorm_conv2d(
        &self,
        kernels: &Self::Output,
        mean: &Self::Output,
        var: &Self::Output,
        gamma: &Self::Output,
        beta: &Self::Output,
        bias: Option<&Self::Output>,
        eps: T,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        algo: Option<ConvAlgo>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        let conv_res = self.conv2d(kernels, bias, steps, padding, dilation, algo)?;
        let (kernel, reg_info) = load_ptx_and_get_data(
            "conv2d_batchnorm",
            &format!("batchnorm_forward_{}", T::STR),
            conv_res.device(),
            conv_res.device_cap(),
            &CONV2D_BATCHNORM,
        )?;
        let cfg = compute_kernel_launch_config(conv_res.device(), &reg_info, conv_res.size());
        let in_out = conv_res.cuda_slice();
        let gamma_slice = gamma.cuda_slice();
        let beta_slice = beta.cuda_slice();
        let mean_slice = mean.cuda_slice();
        let var_slice = var.cuda_slice();
        let size = conv_res.size();
        let channels = conv_res.shape()[3] as usize;
        unsafe {
            kernel.launch(
                cfg,
                (
                    in_out,
                    gamma_slice,
                    beta_slice,
                    mean_slice,
                    var_slice,
                    eps,
                    size,
                    channels,
                ),
            )
        }?;
        Ok(conv_res)
    }
}
