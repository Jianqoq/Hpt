use crate::{Cpu, Tensor};
use tensor_common::error::base::TensorError;
use tensor_traits::ops::advance::{AdvanceOps, HardMax, Shrinkage};
use tensor_traits::CommonBounds;
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::into_vec::IntoVec;
use tensor_types::traits::SimdSelect;
use tensor_types::type_promote::{Cmp, NormalOut, SimdCmp};

impl<T: CommonBounds + PartialOrd, const DEVICE: usize> AdvanceOps for Tensor<T, Cpu, DEVICE>
where
    T: NormalOut<bool, Output = T>,
    f64: IntoScalar<T>,
{
    type Meta = T;
    type Output = Tensor<T, Cpu, DEVICE>;

    type IndexOutput = Tensor<i64, Cpu, DEVICE>;

    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError> {
        Ok(self.inner.pad(pads, val)?.into())
    }

    fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError> {
        let (indices, values) = self.inner.topk(k, dim, largest, sorted)?;
        Ok((indices.into(), values.into()))
    }

    fn onehot(
        &self,
        depth: usize,
        axis: i64,
        true_val: Self::Meta,
        false_val: Self::Meta,
    ) -> Result<Self::Output, TensorError> {
        Ok(self.inner.onehot(depth, axis, true_val, false_val)?.into())
    }

    fn gather(&self, indices: &Self::IndexOutput, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.gather(indices.inner.as_ref(), axis)?.into())
    }

    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.dropout(rate)?.into())
    }
    
    fn gather_elements(&self, indices: &Self::IndexOutput, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.gather_elements(indices.inner.as_ref(), axis)?.into())
    }
}

impl<T: CommonBounds, const DEVICE: usize> Shrinkage<T> for Tensor<T, Cpu, DEVICE>
where
    T: Cmp<Output = bool> + TypeCommon,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
{
    type Output = Tensor<T, Cpu, DEVICE>;
    fn shrinkage(&self, bias: T, lambda: T) -> Result<Self::Output, TensorError> {
        Ok(self.inner.shrinkage(bias, lambda)?.into())
    }
}

impl<T, const DEVICE: usize> HardMax<T> for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + Cmp<Output = bool>,
    <T as TypeCommon>::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: IntoVec<T::Vec>,
    bool: NormalOut<T> + IntoScalar<T>,
{
    type Output = Tensor<T, Cpu, DEVICE>;
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.hardmax(axis)?.into())
    }
}
