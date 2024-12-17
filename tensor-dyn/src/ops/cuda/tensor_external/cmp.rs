#![allow(unused)]

use std::borrow::Borrow;

use crate::ops::cuda::cuda_utils::cast_operand;
use crate::{ops::cuda::binary_normal::binary_fn_with_out_simd, Cuda};
use crate::{tensor::Tensor, tensor_base::_Tensor, BoolVector};
use anyhow::Result;
use cudarc::driver::DeviceRepr;
use tensor_traits::{tensor::CommonBounds, TensorCmp};
use tensor_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{Cmp, SimdCmp},
};

impl<T, C, const DEVICE_ID: usize> TensorCmp<T, C> for Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + DeviceRepr,
    C: CommonBounds + DeviceRepr,
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        T::Vec: SimdCmp<C::Vec>,
        <T::Vec as SimdCmp<C::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    "!=",
                    cast_operand::<bool, C>(y),
                )
            },
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        <T as TypeCommon>::Vec: SimdCmp<<C as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<C as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    "==",
                    cast_operand::<bool, C>(y),
                )
            },
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        <T as TypeCommon>::Vec: SimdCmp<<C as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<C as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    "<",
                    cast_operand::<bool, C>(y),
                )
            },
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        <T as TypeCommon>::Vec: SimdCmp<<C as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<C as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    ">",
                    cast_operand::<bool, C>(y),
                )
            },
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        <T as TypeCommon>::Vec: SimdCmp<<C as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<C as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    "<=",
                    cast_operand::<bool, C>(y),
                )
            },
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
        T: Cmp<C>,
        D: Borrow<Tensor<C, Cuda, DEVICE_ID>>,
        <T as TypeCommon>::Vec: SimdCmp<<C as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp<<C as TypeCommon>::Vec>>::Output: IntoVec<BoolVector>,
    {
        let res = binary_fn_with_out_simd(
            self.inner.as_ref(),
            rhs.borrow().inner.as_ref(),
            |out, x, y| {
                format!(
                    "{out} = {} {} {}",
                    cast_operand::<bool, T>(x),
                    ">=",
                    cast_operand::<bool, C>(y),
                )
            },
            None::<_Tensor<bool, Cuda, DEVICE_ID>>,
        )?;
        Ok(res.into())
    }
}
