#![allow(unused)]

use cudarc::{
    cudnn::{ConvForward, Cudnn, CudnnDataType},
    driver::{CudaSlice, DeviceRepr},
};
use hpt_common::{error::shape::ShapeError, utils::conv_algos::ConvAlgo};
use hpt_traits::ops::{
    creation::TensorCreator, shape_manipulate::ShapeManipulate, unary::Contiguous,
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

use crate::{backends::common::conv::cal_conv2d_output_shape, tensor_base::_Tensor, CUDNN};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};

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
        ShapeError::check_contiguous(
            "Conv2d requires kernel tensor to be contiguous. ".to_string(),
            kernels.layout(),
        )?;
        let res_device = self.device();
        let cudnn = Cudnn::new(res_device).unwrap(); // when drop in test env, it throws error STATUS_STACK_BUFFER_OVERRUN, for now, I have to use this new
        let inp_shape = [
            self.shape()[0] as i32,
            self.shape()[1] as i32,
            self.shape()[2] as i32,
            self.shape()[3] as i32,
        ];
        let kernel_shape = [
            kernels.shape()[0] as i32,
            kernels.shape()[1] as i32,
            kernels.shape()[2] as i32,
            kernels.shape()[3] as i32,
        ];
        let (step_width, step_height) = (steps[0], steps[1]);
        let (ph, pw) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let (out_height, out_width) = cal_conv2d_output_shape(
            inp_shape[1] as i64,
            inp_shape[2] as i64,
            kernels.shape()[0] as i64,
            kernels.shape()[1] as i64,
            &[(ph, ph), (pw, pw)],
            &[step_height, step_width],
            &[dh, dw],
        );

        let output_shape = [
            inp_shape[0] as i32,
            out_height as i32,
            out_width as i32,
            kernel_shape[3] as i32,
        ];

        let conv2d = cudnn
            .create_conv2d::<T>(
                [padding[0] as i32, padding[1] as i32],
                [steps[0] as i32, steps[1] as i32],
                [dilation[0] as i32, dilation[1] as i32],
                cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .expect("create conv2d failed");
        let x = if self.is_contiguous() && self.parent().is_none() {
            cudnn
                .create_4d_tensor::<T>(
                    cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                    [inp_shape[0], inp_shape[3], inp_shape[1], inp_shape[2]],
                )
                .expect("create tensor failed")
        } else {
            let strides = self.strides();
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
                    kernel_shape[3],
                    kernel_shape[2],
                    kernel_shape[0],
                    kernel_shape[1],
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
        let mut workspace = self.device().alloc_zeros::<u8>(workspace_size)?;

        let res = Self::Output::empty(&output_shape)?;
        let mut res_slice = unsafe {
            res.device()
                .upgrade_device_ptr::<T>(res.ptr().ptr as u64, res.size())
        };
        let inp_slice = unsafe {
            self.device()
                .upgrade_device_ptr::<T>(self.ptr().ptr as u64, self.size())
        };
        let kernels = kernels.permute([3, 0, 1, 2])?.contiguous()?;
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
        if let Some(bias) = bias {
            return Ok(bias + res);
        }
        res_slice.leak();
        kernel_slice.leak();
        inp_slice.leak();
        Ok(res)
    }

    fn conv2d_group(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        groups: i64,
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        unimplemented!()
    }

    fn dwconv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        unimplemented!()
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

impl<T, const DEVICE: usize, A> ConvBatchNorm<T> for _Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
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
        padding: [(i64, i64); 2],
        dilation: [i64; 2],
        activation: Option<fn(<T>::Vec) -> <T>::Vec>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        unimplemented!()
    }
}
