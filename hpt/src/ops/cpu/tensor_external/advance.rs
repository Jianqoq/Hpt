use std::cell::RefCell;
use std::rc::Rc;

use crate::ops::cpu::utils::diff::diff_utils::handle_grad;
use crate::tensor::DiffTensor;
use crate::tensor_base::_Tensor;
use crate::{Cpu, Tensor};
use hpt_common::error::base::TensorError;
use hpt_common::slice::Slice;
use hpt_traits::ops::advance::{AdvancedOps, HardMax, Shrinkage};
use hpt_traits::{CommonBounds, TensorCreator, TensorInfo, TensorWhere};
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::into_vec::IntoVec;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::{Cmp, NormalOut, SimdCmp};

impl<T: CommonBounds + PartialOrd, const DEVICE: usize> AdvancedOps for Tensor<T, Cpu, DEVICE>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
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

    // fn gather(&self, indices: &Self::IndexOutput, axis: i64) -> Result<Self::Output, TensorError> {
    //     Ok(self.inner.gather(indices.inner.as_ref(), axis)?.into())
    // }

    fn dropout(&self, rate: f64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.dropout(rate)?.into())
    }

    // fn gather_elements(
    //     &self,
    //     indices: &Self::IndexOutput,
    //     axis: i64,
    // ) -> Result<Self::Output, TensorError> {
    //     Ok(self
    //         .inner
    //         .gather_elements(indices.inner.as_ref(), axis)?
    //         .into())
    // }

    fn scatter(
        &self,
        indices: &Self::IndexOutput,
        axis: i64,
        src: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        Ok(self
            .inner
            .scatter(indices.inner.as_ref(), axis, src.inner.as_ref())?
            .into())
    }
}

impl<T: CommonBounds + PartialOrd, const DEVICE: usize> AdvancedOps for DiffTensor<T, Cpu, DEVICE>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
{
    type Meta = T;
    type Output = DiffTensor<T, Cpu, DEVICE>;

    type IndexOutput = Tensor<i64, Cpu, DEVICE>;

    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError> {
        let padded = self.inner.pad(pads, val)?;
        let pads = pads.to_vec();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: padded,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE>| {
                let mut ranges = Vec::with_capacity(pads.len());

                for (dim, (pad_before, pad_after)) in pads.iter().enumerate() {
                    ranges.push(Slice::Range((*pad_before, grad.shape()[dim] - *pad_after)));
                }
                grad = grad.slice(&ranges)?;
                handle_grad(&mut lhs, grad, &[])?;
                Ok(false)
            })),
        })
    }

    fn topk(
        &self,
        k: i64,
        dim: i64,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self::IndexOutput, Self::Output), TensorError> {
        let (indices, values) = self.inner.topk(k, dim, largest, sorted)?;
        let mut lhs = self.clone();
        Ok((
            indices.clone(),
            DiffTensor {
                inner: values,
                grad: Rc::new(RefCell::new(None)),
                out_degree: Rc::new(RefCell::new(0)),
                backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE>| {
                    let full_grad = Tensor::<T, Cpu, DEVICE>::zeros(lhs.inner.shape())?;
                    full_grad.scatter(&indices, dim, &grad)?;
                    handle_grad(&mut lhs, full_grad, &[])?;
                    Ok(false)
                })),
            },
        ))
    }

    fn onehot(
        &self,
        _: usize,
        _: i64,
        _: Self::Meta,
        _: Self::Meta,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }

    // fn gather(&self, _: &Self::IndexOutput, _: i64) -> Result<Self::Output, TensorError> {
    //     unimplemented!()
    // }

    fn dropout(&self, _: f64) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }

    // fn gather_elements(&self, _: &Self::IndexOutput, _: i64) -> Result<Self::Output, TensorError> {
    //     unimplemented!()
    // }

    fn scatter(
        &self,
        _: &Self::IndexOutput,
        _: i64,
        _: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
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
    bool: NormalOut<T> + Cast<T>,
{
    type Output = Tensor<T, Cpu, DEVICE>;
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.hardmax(axis)?.into())
    }
}

impl<T, const DEVICE: usize> TensorWhere for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds,
{
    type Output = Tensor<T, Cpu, DEVICE>;
    type Condition = Tensor<bool, Cpu, DEVICE>;
    fn tensor_where(
        condition: &Self::Condition,
        x: &Self::Output,
        y: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        let res = _Tensor::<T, Cpu, DEVICE>::tensor_where(
            condition.inner.as_ref(),
            x.inner.as_ref(),
            y.inner.as_ref(),
        )?;
        Ok(res.into())
    }
}
