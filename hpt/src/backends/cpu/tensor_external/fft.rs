use crate::tensor::Tensor;
use hpt_common::axis::axis::Axis;
use hpt_common::error::base::TensorError;
use hpt_common::shape::shape::Shape;
use hpt_traits::ops::fft::FFTOps;
use num::complex::Complex32;
use num::complex::Complex64;
use std::sync::Arc;

impl FFTOps for Tensor<Complex32> {
    fn fft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
        self.fftn(&[n], axis, norm)
    }
    fn ifft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
        self.ifftn(&[n], axis, norm)
    }
    fn fft2<S: Into<Shape>>(
        &self,
        s: S,
        axis1: i64,
        axis2: i64,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        self.fftn(s, [axis1, axis2], norm)
    }
    fn ifft2<S: Into<Shape>>(
        &self,
        s: S,
        axis1: i64,
        axis2: i64,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        self.ifftn(s, [axis1, axis2], norm)
    }
    fn fftn<A: Into<Axis>, S: Into<Shape>>(
        &self,
        s: S,
        axes: A,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        let inner = self.inner.fftn(s, axes, norm)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    fn ifftn<A: Into<Axis>, S: Into<Shape>>(
        &self,
        s: S,
        axes: A,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        let inner = self.inner.ifftn(s, axes, norm)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl FFTOps for Tensor<Complex64> {
    fn fft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
        self.fftn(&[n], axis, norm)
    }
    fn ifft(&self, n: usize, axis: i64, norm: Option<&str>) -> Result<Self, TensorError> {
        self.ifftn(&[n], axis, norm)
    }
    fn fft2<S: Into<Shape>>(
        &self,
        s: S,
        axis1: i64,
        axis2: i64,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        self.fftn(s, [axis1, axis2], norm)
    }
    fn ifft2<S: Into<Shape>>(
        &self,
        s: S,
        axis1: i64,
        axis2: i64,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        self.ifftn(s, [axis1, axis2], norm)
    }
    fn fftn<A: Into<Axis>, S: Into<Shape>>(
        &self,
        s: S,
        axes: A,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        let inner = self.inner.fftn(s, axes, norm)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    fn ifftn<A: Into<Axis>, S: Into<Shape>>(
        &self,
        s: S,
        axes: A,
        norm: Option<&str>,
    ) -> Result<Self, TensorError> {
        let inner = self.inner.ifftn(s, axes, norm)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}
