#![allow(unused)]

use std::borrow::Borrow;

use crate::backends::cuda::utils::binary::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cuda;
use hpt_common::error::base::TensorError;
use hpt_traits::{ops::cmp::TensorCmp, tensor::CommonBounds};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE_ID: usize, Al> TensorCmp<T, C> for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + DeviceRepr + Cmp<C> + CudaType,
    C: CommonBounds + DeviceRepr + CudaType,
    Scalar<T>: Cmp<Scalar<C>, Output = Scalar<bool>>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type RHS = Tensor<C, Cuda, DEVICE_ID, Al>;
    type Output = Tensor<bool, Cuda, DEVICE_ID, Al>;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "neq",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._ne(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "eq",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._eq(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "lt",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._lt(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "gt",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._gt(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "le",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._le(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Self::Output, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            "ge",
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._ge(y)),
            None::<Self::Output>,
        )?;
        Ok(res.into())
    }
}
