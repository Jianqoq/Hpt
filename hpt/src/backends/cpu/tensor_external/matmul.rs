use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use hpt_common::error::base::TensorError;
use hpt_traits::{
    ops::{binary::Matmul, shape_manipulate::ShapeManipulate},
    tensor::CommonBounds,
};

use crate::{
    backends::cpu::{
        kernels::matmul::microkernel_trait::MatmulMicroKernel, utils::diff::diff_utils::handle_grad,
    },
    tensor::{DiffTensor, Tensor},
};
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};

impl<T, const DEVICE: usize, Al> Matmul<Tensor<T, Cpu, DEVICE, Al>> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + MatmulMicroKernel,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: Tensor<T, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.matmul(rhs.inner.as_ref())?.into())
    }
    fn matmul_<U>(
        &self,
        rhs: Tensor<T, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(self.inner.matmul_(rhs.inner.as_ref(), out)?.into())
    }
}

impl<T, const DEVICE: usize, Al> Matmul<&Tensor<T, Cpu, DEVICE, Al>> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + MatmulMicroKernel,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: &Tensor<T, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(self.inner.matmul(rhs.inner.as_ref())?.into())
    }

    fn matmul_<U>(
        &self,
        rhs: &Tensor<T, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::Output, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        let out = out.borrow().inner.as_ref().clone();
        Ok(self.inner.matmul_(rhs.inner.as_ref(), out)?.into())
    }
}

impl<T, const DEVICE: usize, Al> Matmul<DiffTensor<T, Cpu, DEVICE, Al>>
    for DiffTensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + MatmulMicroKernel,
    Al: Allocator + 'static + Send + Sync,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = DiffTensor<T, Cpu, DEVICE, Al>;

    type OutputMeta = T;

    type InplaceOutput = Tensor<T, Cpu, DEVICE, Al>;

    fn matmul(
        &self,
        rhs: DiffTensor<T, Cpu, DEVICE, Al>,
    ) -> std::result::Result<Self::Output, TensorError> {
        let res = self.inner.matmul(&rhs.inner)?;
        let mut lhs = self.clone();
        let mut rhs = rhs.clone();
        Ok(DiffTensor {
            inner: res,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE, Al>| {
                let grad_a = grad.matmul(rhs.inner.t()?)?;
                let grad_b = lhs.inner.t()?.matmul(grad)?;
                handle_grad(&mut lhs, grad_a, &[])?;
                handle_grad(&mut rhs, grad_b, &[])?;
                Ok(false)
            })),
        })
    }
    fn matmul_<U>(
        &self,
        rhs: DiffTensor<T, Cpu, DEVICE, Al>,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: Borrow<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>,
    {
        self.inner.matmul_(&rhs.inner, out)
    }
}
