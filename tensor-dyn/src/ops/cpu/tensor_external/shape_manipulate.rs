use crate::ops::cpu::concat::concat;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use tensor_common::error::base::TensorError;
use tensor_common::{axis::Axis, shape::shape::Shape};
use tensor_traits::{CommonBounds, ShapeManipulate};

impl<T: CommonBounds> ShapeManipulate for Tensor<T> {
    type Meta = T;
    type Output = Tensor<T>;

    fn squeeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::squeeze(self.inner.as_ref(), axes)?.into())
    }

    fn unsqueeze<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::unsqueeze(self.inner.as_ref(), axes)?.into())
    }

    fn reshape<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::reshape(self.inner.as_ref(), shape)?.into())
    }

    fn transpose(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::transpose(self.inner.as_ref(), axis1, axis2)?.into())
    }

    fn permute<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::permute(self.inner.as_ref(), axes)?.into())
    }

    fn permute_inv<A: Into<Axis>>(
        &self,
        axes: A,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::permute_inv(self.inner.as_ref(), axes)?.into())
    }

    fn expand<S: Into<Shape>>(&self, shape: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::expand(self.inner.as_ref(), shape)?.into())
    }

    fn t(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::t(self.inner.as_ref())?.into())
    }

    fn mt(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::mt(self.inner.as_ref())?.into())
    }

    fn flip<A: Into<Axis>>(&self, axes: A) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::flip(self.inner.as_ref(), axes)?.into())
    }

    fn fliplr(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::fliplr(self.inner.as_ref())?.into())
    }

    fn flipud(&self) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::flipud(self.inner.as_ref())?.into())
    }

    fn tile<S: Into<Axis>>(&self, repeats: S) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::tile(self.inner.as_ref(), repeats)?.into())
    }

    fn trim_zeros(&self, trim: &str) -> std::result::Result<Self::Output, TensorError>
    where
        Self::Meta: PartialEq,
    {
        Ok(_Tensor::trim_zeros(self.inner.as_ref(), trim)?.into())
    }

    fn repeat(&self, repeats: usize, axes: i16) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::repeat(self.inner.as_ref(), repeats, axes)?.into())
    }

    fn split(
        &self,
        indices_or_sections: &[i64],
        axis: i64,
    ) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(
            _Tensor::split(self.inner.as_ref(), indices_or_sections, axis)?
                .into_iter()
                .map(|x| x.into())
                .collect(),
        )
    }

    fn dsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(_Tensor::dsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn hsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(_Tensor::hsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn vsplit(&self, indices: &[i64]) -> std::result::Result<Vec<Self>, TensorError> {
        Ok(_Tensor::vsplit(self.inner.as_ref(), indices)?
            .into_iter()
            .map(|x| x.into())
            .collect())
    }

    fn swap_axes(&self, axis1: i64, axis2: i64) -> std::result::Result<Self::Output, TensorError> {
        Ok(_Tensor::swap_axes(self.inner.as_ref(), axis1, axis2)?.into())
    }

    fn flatten<A>(&self, start: A, end: A) -> std::result::Result<Self::Output, TensorError>
    where
        A: Into<Option<usize>>,
    {
        Ok(_Tensor::flatten(self.inner.as_ref(), start, end)?.into())
    }

    fn concat(
        tensors: Vec<&Self>,
        axis: usize,
        keepdims: bool,
    ) -> std::result::Result<Self::Output, TensorError> {
        Ok(concat(
            tensors.iter().map(|x| x.inner.as_ref()).collect(),
            axis,
            keepdims,
        )?
        .into())
    }

    fn vstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(concat(tensors.iter().map(|x| x.inner.as_ref()).collect(), 0, false)?.into())
    }

    fn hstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(concat(tensors.iter().map(|x| x.inner.as_ref()).collect(), 1, false)?.into())
    }

    fn dstack(tensors: Vec<&Self>) -> std::result::Result<Self::Output, TensorError> {
        Ok(concat(tensors.iter().map(|x| x.inner.as_ref()).collect(), 2, false)?.into())
    }
}
