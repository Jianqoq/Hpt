use std::borrow::{Borrow, BorrowMut};

use cudarc::{
    cublas::{CudaBlas, Gemm},
    driver::DeviceRepr,
};
use tensor_traits::{CommonBounds, Matmul};

use crate::{ops::cuda::matmul::matmul_with_out, tensor::Tensor, Cuda};

impl<T, const CUDA_DEVICE: usize> Matmul<Tensor<T, Cuda, CUDA_DEVICE>>
    for Tensor<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + DeviceRepr,
    CudaBlas: Gemm<T>,
{
    type Output = Tensor<T, Cuda, CUDA_DEVICE>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cuda, CUDA_DEVICE>;

    fn matmul(&self, rhs: Tensor<T, Cuda, CUDA_DEVICE>) -> anyhow::Result<Self::Output> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }
    fn matmul_<U>(&self, rhs: Tensor<T, Cuda, CUDA_DEVICE>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}

impl<T, const CUDA_DEVICE: usize> Matmul<&Tensor<T, Cuda, CUDA_DEVICE>>
    for Tensor<T, Cuda, CUDA_DEVICE>
where
    T: CommonBounds + DeviceRepr,
    CudaBlas: Gemm<T>,
{
    type Output = Tensor<T, Cuda, CUDA_DEVICE>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cuda, CUDA_DEVICE>;

    fn matmul(&self, rhs: &Tensor<T, Cuda, CUDA_DEVICE>) -> anyhow::Result<Self::Output> {
        Ok(matmul_with_out(
            self.inner.as_ref(),
            rhs.inner.as_ref(),
            None::<Self::Output>,
        )?
        .into())
    }

    fn matmul_<U>(&self, rhs: &Tensor<T, Cuda, CUDA_DEVICE>, out: U) -> anyhow::Result<Self::Output>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(matmul_with_out(self.inner.as_ref(), rhs.inner.as_ref(), Some(out))?.into())
    }
}
