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
use hpt_traits::ops::binary::Gemm as HptGemm;
use hpt_traits::tensor::CommonBounds;
use hpt_types::dtype::CudaType;
impl<T, const CUDA_DEVICE: usize, Al> HptGemm<Tensor<T, Cuda, CUDA_DEVICE, Al>>
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

    fn gemm(
        &self,
        rhs: Tensor<T, Cuda, CUDA_DEVICE, Al>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        if conj_dst || conj_lhs || conj_rhs {
            panic!("conj_dst, conj_lhs, conj_rhs is not supported for cuda gemm yet");
        }
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            alpha,
            beta,
            None::<Self::Output>,
        )?
        .into())
    }
    fn gemm_<U>(
        &self,
        rhs: Tensor<T, Cuda, CUDA_DEVICE, Al>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        if conj_dst || conj_lhs || conj_rhs {
            panic!("conj_dst, conj_lhs, conj_rhs is not supported for cuda gemm yet");
        }
        let out = out.borrow().inner.as_ref().clone();
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            alpha,
            beta,
            Some(out),
        )?
        .into())
    }
}

impl<T, const CUDA_DEVICE: usize, Al> HptGemm<&Tensor<T, Cuda, CUDA_DEVICE, Al>>
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

    fn gemm(
        &self,
        rhs: &Tensor<T, Cuda, CUDA_DEVICE, Al>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        if conj_dst || conj_lhs || conj_rhs {
            panic!("conj_dst, conj_lhs, conj_rhs is not supported for cuda gemm yet");
        }
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            alpha,
            beta,
            None::<Self::Output>,
        )?
        .into())
    }

    fn gemm_<U>(
        &self,
        rhs: &Tensor<T, Cuda, CUDA_DEVICE, Al>,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        if conj_dst || conj_lhs || conj_rhs {
            panic!("conj_dst, conj_lhs, conj_rhs is not supported for cuda gemm yet");
        }
        let out = out.borrow().inner.as_ref().clone();
        Ok(gemm_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            alpha,
            beta,
            Some(out),
        )?
        .into())
    }
}
