use std::cell::RefCell;
use std::rc::Rc;

use crate::backends::cpu::utils::diff::diff_utils::handle_grad;
use crate::tensor::DiffTensor;
use crate::tensor_base::_Tensor;
use crate::Tensor;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_allocator::Cpu;
use hpt_common::error::base::TensorError;
use hpt_traits::ops::advance::{AdvancedOps, HardMax, Shrinkage, TensorWhere};
use hpt_traits::ops::creation::TensorCreator;
use hpt_traits::ops::slice::Slice;
use hpt_traits::tensor::{CommonBounds, TensorInfo};
use hpt_types::dtype::TypeCommon;
use hpt_types::into_scalar::Cast;
use hpt_types::into_vec::IntoVec;
use hpt_types::traits::SimdSelect;
use hpt_types::type_promote::{Cmp, NormalOut, SimdCmp};

impl<T: CommonBounds + PartialOrd, const DEVICE: usize, Al> AdvancedOps
    for Tensor<T, Cpu, DEVICE, Al>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = Tensor<T, Cpu, DEVICE, Al>;

    type IndexOutput = Tensor<i64, Cpu, DEVICE, Al>;

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

impl<T: CommonBounds + PartialOrd, const DEVICE: usize, Al> AdvancedOps
    for DiffTensor<T, Cpu, DEVICE, Al>
where
    T: NormalOut<bool, Output = T> + Cast<i64>,
    f64: Cast<T>,
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = DiffTensor<T, Cpu, DEVICE, Al>;

    type IndexOutput = Tensor<i64, Cpu, DEVICE, Al>;

    fn pad(&self, pads: &[(i64, i64)], val: Self::Meta) -> Result<Self::Output, TensorError> {
        let padded = self.inner.pad(pads, val)?;
        let pads = pads.to_vec();
        let mut lhs = self.clone();
        Ok(DiffTensor {
            inner: padded,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |mut grad: Tensor<T, Cpu, DEVICE, Al>| {
                let mut ranges = Vec::with_capacity(pads.len());

                for (dim, (pad_before, pad_after)) in pads.iter().enumerate() {
                    ranges.push((*pad_before, grad.shape()[dim] - *pad_after, 1));
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
                backward: Rc::new(RefCell::new(move |grad: Tensor<T, Cpu, DEVICE, Al>| {
                    let full_grad = Tensor::<T, Cpu, DEVICE, Al>::zeros(lhs.inner.shape())?;
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

    fn scatter(
        &self,
        _: &Self::IndexOutput,
        _: i64,
        _: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        unimplemented!()
    }
}

impl<T: CommonBounds, const DEVICE: usize, Al> Shrinkage<T> for Tensor<T, Cpu, DEVICE, Al>
where
    T: Cmp<Output = bool> + TypeCommon,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: SimdSelect<T::Vec>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;
    fn shrinkage(&self, bias: T, lambda: T) -> Result<Self::Output, TensorError> {
        Ok(self.inner.shrinkage(bias, lambda)?.into())
    }
}

impl<T, const DEVICE: usize, Al> HardMax<T> for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds + Cmp<Output = bool>,
    <T as TypeCommon>::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: IntoVec<T::Vec>,
    bool: NormalOut<T> + Cast<T>,
    Al: Allocator + Send + Sync + 'static,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;
    fn hardmax(&self, axis: i64) -> Result<Self::Output, TensorError> {
        Ok(self.inner.hardmax(axis)?.into())
    }
}

impl<T, const DEVICE: usize, Al> TensorWhere for Tensor<T, Cpu, DEVICE, Al>
where
    T: CommonBounds,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cpu, DEVICE, Al>;
    type Condition = Tensor<bool, Cpu, DEVICE, Al>;
    fn tensor_where(
        condition: &Self::Condition,
        x: &Self::Output,
        y: &Self::Output,
    ) -> Result<Self::Output, TensorError> {
        let res = _Tensor::<T, Cpu, DEVICE, Al>::tensor_where(
            condition.inner.as_ref(),
            x.inner.as_ref(),
            y.inner.as_ref(),
        )?;
        Ok(res.into())
    }
}
