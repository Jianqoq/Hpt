#![allow(unused)]

use std::borrow::Borrow;

use crate::{ops::cuda::binary_normal::binary_fn_with_out_simd, Cuda};
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use hpt_traits::{tensor::CommonBounds, TensorCmp};
use hpt_types::cuda_types::scalar::Scalar;
use hpt_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE_ID: usize> TensorCmp<T, C> for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr + Cmp<C>,
    C: CommonBounds + DeviceRepr,
    T::Vec: SimdCmp<C::Vec>,
    <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
    Scalar<T>: Cmp<Scalar<C>, Output = Scalar<bool>>,
{
    type RHS = Tensor<C, Cuda, DEVICE_ID>;
    type Output = Tensor<bool, Cuda, DEVICE_ID>;
    type BoolVector = BoolVector;
    /// perform element-wise not equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_neq<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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

    /// perform element-wise equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_eq<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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

    /// perform element-wise less than operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_lt<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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

    /// perform element-wise greater than operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_gt<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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

    /// perform element-wise less than or equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_le<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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

    /// perform element-wise greater than or equal operation between two tensors
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right hand side tensor
    ///
    /// # Returns
    ///
    /// A tensor of boolean values
    fn tensor_ge<D>(&self, rhs: D) -> Result<Tensor<bool, Cuda, DEVICE_ID>>
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
