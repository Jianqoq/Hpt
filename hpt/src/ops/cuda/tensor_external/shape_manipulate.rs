use crate::Cuda;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::error::base::TensorError;
use hpt_common::{axis::axis::Axis, shape::shape::Shape};
use hpt_traits::{CommonBounds, Concat, ShapeManipulate};
use hpt_types::dtype::CudaType;

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE: usize, Al> ShapeManipulate
    for Tensor<T, Cuda, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;
    type Output = Tensor<T, Cuda, DEVICE, Al>;

    fn squeeze<A: Into<Axis>>(&self, axes: A) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::squeeze(self.inner.as_ref(), axes)?.into())
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::unsqueeze(self.inner.as_ref(), axes)?.into())
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::reshape(self.inner.as_ref(), shape)?.into())
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::transpose(self.inner.as_ref(), axis1, axis2)?.into())
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::permute(self.inner.as_ref(), axes)?.into())
    }

    fn permute_inv<A: Into<Axis>>(&self, axes: A) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::permute_inv(self.inner.as_ref(), axes)?.into())
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::expand(self.inner.as_ref(), shape)?.into())
    }

    fn t(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::t(self.inner.as_ref())?.into())
    }

    fn mt(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::mt(self.inner.as_ref())?.into())
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::flip(self.inner.as_ref(), axes)?.into())
    }

    fn fliplr(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::fliplr(self.inner.as_ref())?.into())
    }

    fn flipud(&self) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::flipud(self.inner.as_ref())?.into())
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::tile(self.inner.as_ref(), repeats)?.into())
    }

    fn trim_zeros(&self, trim: &str) -> Result<Self::Output, TensorError>
    where
        Self::Meta: PartialEq,
    {
        Ok(_Tensor::trim_zeros(self.inner.as_ref(), trim)?.into())
    }

    fn repeat(&self, repeats: usize, axes: i16) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::repeat(self.inner.as_ref(), repeats, axes)?.into())
    }

    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> Result<Vec<Self::Output>, TensorError> {
        Ok(
            _Tensor::split(self.inner.as_ref(), indices_or_sections, axis)?
                .into_iter()
                .map(|x| x.into())
                .collect(),
        )
    }

    fn dsplit(&self, indices: &[i64]) -> Result<Vec<Self::Output>, TensorError> {
        Ok(_Tensor::dsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn hsplit(&self, indices: &[i64]) -> Result<Vec<Self::Output>, TensorError> {
        Ok(_Tensor::hsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn vsplit(&self, indices: &[i64]) -> Result<Vec<Self::Output>, TensorError> {
        Ok(_Tensor::vsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn swap_axes(&self, axis1: i64, axis2: i64) -> Result<Self::Output, TensorError> {
        Ok(_Tensor::swap_axes(self.inner.as_ref(), axis1, axis2)?.into())
    }

    fn flatten<A>(&self, start: A, end: A) -> Result<Self::Output, TensorError>
    where
        A: Into<Option<usize>>,
    {
        Ok(_Tensor::flatten(self.inner.as_ref(), start, end)?.into())
    }
}

impl<T: CommonBounds + DeviceRepr + CudaType, const DEVICE: usize, Al> Concat
    for Tensor<T, Cuda, DEVICE, Al>
where
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Output = Tensor<T, Cuda, DEVICE, Al>;
    fn concat(
        tensors: Vec<Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::concat(
            tensors
                .into_iter()
                .map(|x| x.inner.as_ref().clone())
                .collect(),
            axis,
            keepdims,
        )?
        .into())
    }

    fn vstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::vstack(
            tensors
                .into_iter()
                .map(|x| x.inner.as_ref().clone())
                .collect(),
        )?
        .into())
    }

    fn hstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::hstack(
            tensors
                .into_iter()
                .map(|x| x.inner.as_ref().clone())
                .collect(),
        )?
        .into())
    }

    fn dstack(tensors: Vec<Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::dstack(
            tensors
                .into_iter()
                .map(|x| x.inner.as_ref().clone())
                .collect(),
        )?
        .into())
    }
}
