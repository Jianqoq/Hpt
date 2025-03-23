#![allow(unused)]

use cudarc::{
    cudnn::{ConvForward, Cudnn, CudnnDataType},
    driver::{CudaSlice, DeviceRepr},
};
use hpt_common::{error::shape::ShapeError, utils::conv_algos::ConvAlgo};
use hpt_traits::ops::creation::TensorCreator;
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

use crate::{backends::common::conv::cal_conv2d_output_shape, tensor_base::_Tensor, Tensor, CUDNN};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cuda,
};

impl<T, const DEVICE: usize, Al> CudaConv<T> for Tensor<T, Cuda, DEVICE, Al>
where
    T: CommonBounds + DeviceRepr + CudaType + CudnnDataType,
    bool: Cast<T>,
    Al: Allocator + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
    Scalar<T>: NormalOut<Scalar<T>, Output = Scalar<T>>,
{
    type Output = Tensor<T, Cuda, DEVICE, Al>;

    fn conv2d(
        &self,
        kernels: &Self::Output,
        bias: Option<&Self::Output>,
        steps: [i64; 2],
        padding: [i64; 2],
        dilation: [i64; 2],
        algo: Option<ConvAlgo>,
    ) -> Result<Self::Output, hpt_common::error::base::TensorError> {
        Ok(self
            .inner
            .conv2d(
                kernels.inner.as_ref(),
                bias.map(|b| b.inner.as_ref()),
                steps,
                padding,
                dilation,
                algo,
            )?
            .into())
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

impl<T, const DEVICE: usize, A> ConvBatchNorm<T> for Tensor<T, Cuda, DEVICE, A>
where
    T: CommonBounds,
    T::Vec: FloatOutBinary<Output = T::Vec> + FloatOutUnary<Output = T::Vec>,
    T: FloatOutBinary<Output = T> + FloatOutUnary<Output = T>,
    bool: Cast<T>,
    A: Allocator + Send + Sync,
    A::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cuda, DEVICE, A>;
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
