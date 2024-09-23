use num::complex::Complex32;
use num::complex::Complex64;
use tensor_traits::ops::fft::FFTOps;
use crate::tensor::Tensor;
use tensor_common::axis::Axis;
use std::sync::Arc;

impl FFTOps for Tensor<Complex32> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft(&self, axis: i64) -> anyhow::Result<Self> {
        self.fftn(axis)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft(&self, axis: i64) -> anyhow::Result<Self> {
        self.ifftn(axis)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
        self.fftn([axis1, axis2])
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
        self.ifftn([axis1, axis2])
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
        let inner = self.inner.fftn(axes)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
        let inner = self.inner.ifftn(axes)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl FFTOps for Tensor<Complex64> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft(&self, axis: i64) -> anyhow::Result<Self> {
        self.fftn(axis)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft(&self, axis: i64) -> anyhow::Result<Self> {
        self.ifftn(axis)
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
        self.fftn([axis1, axis2])
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifft2(&self, axis1: i64, axis2: i64) -> anyhow::Result<Self> {
        self.ifftn([axis1, axis2])
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn fftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
        let inner = self.inner.fftn(axes)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ifftn<A: Into<Axis>>(&self, axes: A) -> anyhow::Result<Self> {
        let inner = self.inner.ifftn(axes)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}
