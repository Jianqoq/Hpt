#![allow(unused)]

use std::borrow::Borrow;

use crate::{ops::cuda::utils::binary::binary_normal::binary_fn_with_out_simd, Cuda};
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use cudarc::driver::DeviceRepr;
use hpt_common::error::base::TensorError;
use hpt_traits::{tensor::CommonBounds, TensorCmp};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::dtype::CudaType;
use hpt_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE_ID: usize> TensorCmp<T, C> for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr + Cmp<C> + CudaType,
    C: CommonBounds + DeviceRepr + CudaType,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
    Scalar<T>: Cmp<Scalar<C>, Output = Scalar<bool>>,
{
    type RHS = Tensor<C, Cuda, DEVICE_ID>;
    type Output = Tensor<bool, Cuda, DEVICE_ID>;
    type BoolVector = BoolVector;

    fn tensor_neq<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._eq(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }

    fn tensor_eq<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._eq(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }

    fn tensor_lt<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._lt(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }

    fn tensor_gt<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._gt(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }

    fn tensor_le<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._le(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }

    fn tensor_ge<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>, TensorError>
    where
        D: Borrow<Self::RHS>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| out.assign(x._ge(y)),
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }
}
