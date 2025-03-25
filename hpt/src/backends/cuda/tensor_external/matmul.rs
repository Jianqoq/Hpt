use std::borrow::{Borrow, BorrowMut};

use crate::{
    backend::Cuda, backends::cuda::tensor_internal::matmul::gemm_with_out, tensor::Tensor,
};
use cudarc::{
    cublas::{CudaBlas, Gemm},
    driver::DeviceRepr,
};
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::binary::Matmul, tensor::CommonBounds};
use hpt_types::dtype::CudaType;
impl<T, const CUDA_DEVICE: usize, Al> Matmul<Tensor<T, Cuda, CUDA_DEVICE, Al>>
    for Tensor<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + DeviceRepr + CudaType,
    CudaBlas: Gemm<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cuda, CUDA_DEVICE, Al>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cuda, CUDA_DEVICE, Al>;

    fn matmul(
        &self,
        rhs: Tensor<T, Cuda, CUDA_DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            T::ONE,
            T::ZERO,
            None::<Self::Output>,
        )?
        .into())
    }
    fn matmul_<U>(
        &self,
        rhs: Tensor<T, Cuda, CUDA_DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            T::ONE,
            T::ZERO,
            Some(out),
        )?
        .into())
    }
}

impl<T, const CUDA_DEVICE: usize, Al> Matmul<&Tensor<T, Cuda, CUDA_DEVICE, Al>>
    for Tensor<T, Cuda, CUDA_DEVICE, Al>
where
    T: CommonBounds + DeviceRepr + CudaType,
    CudaBlas: Gemm<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cuda, CUDA_DEVICE, Al>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cuda, CUDA_DEVICE, Al>;

    fn matmul(
        &self,
        rhs: &Tensor<T, Cuda, CUDA_DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            T::ONE,
            T::ZERO,
            None::<Self::Output>,
        )?
        .into())
    }

    fn matmul_<U>(
        &self,
        rhs: &Tensor<T, Cuda, CUDA_DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            T::ONE,
            T::ZERO,
            Some(out),
        )?
        .into())
    }
}
