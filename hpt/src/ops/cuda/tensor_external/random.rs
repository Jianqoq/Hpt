use crate::{tensor::Tensor, tensor_base::_Tensor, Cuda};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, Random, RandomInt};
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Open01, OpenClosed01, Standard, StandardNormal,
};

impl<T, const DEVICE_ID: usize, Al> Random for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds
        + SampleUniform
        + num::Float
        + rand_distr::num_traits::FloatConst
        + DeviceRepr
        + CudaType,
    <T as SampleUniform>::Sampler: Sync,
    StandardNormal: Distribution<T>,
    Open01: Distribution<T>,
    Exp1: Distribution<T>,
    OpenClosed01: Distribution<T>,
    Standard: Distribution<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::NormalFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::UniformFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::LogNormalFill<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::randn(shape)?.into())
    }

    fn randn_like(&self) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::randn_like(self.inner.as_ref())?.into())
    }

    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::rand(shape, low, high)?.into())
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::rand_like(self.inner.as_ref(), low, high)?.into())
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::beta(a, b, shape)?.into())
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::beta_like(self.inner.as_ref(), a, b)?.into())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::chisquare(df, shape)?.into())
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::chisquare_like(self.inner.as_ref(), df)?.into())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::exponential(lambda, shape)?.into())
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::exponential_like(self.inner.as_ref(), lambda)?
                .into(),
        )
    }

    fn gamma<S: Into<Shape>>(
        gamm_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::gamma(gamm_shape, scale, shape)?.into())
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        Ok(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::gamma_like(self.inner.as_ref(), shape, scale)?
                .into(),
        )
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::gumbel(mu, beta, shape)?.into())
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::gumbel_like(self.inner.as_ref(), mu, beta)?.into())
    }

    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::lognormal(mean, std, shape)?.into())
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::lognormal_like(self.inner.as_ref(), mean, std)?
                .into(),
        )
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::normal_gaussian(mean, std, shape)?.into())
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::normal_gaussian_like(
                self.inner.as_ref(),
                mean,
                std,
            )?
            .into(),
        )
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::pareto(pareto_shape, a, shape)?.into())
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(
            _Tensor::<T, Cuda, DEVICE_ID, Al>::pareto_like(self.inner.as_ref(), pareto_shape, a)?
                .into(),
        )
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::poisson(lambda, shape)?.into())
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::poisson_like(self.inner.as_ref(), lambda)?.into())
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::weibull(a, b, shape)?.into())
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::weibull_like(self.inner.as_ref(), a, b)?.into())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::zipf(n, a, shape)?.into())
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::zipf_like(self.inner.as_ref(), n, a)?.into())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::triangular(low, high, mode, shape)?.into())
    }

    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::triangular_like(
            self.inner.as_ref(),
            low,
            high,
            mode,
        )?
        .into())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::bernoulli(shape, p)?.into())
    }
}

impl<T, const DEVICE_ID: usize, Al> RandomInt for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + SampleUniform,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
{
    type Meta = T;

    fn randint<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::randint(low, high, shape)?.into())
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(_Tensor::<T, Cuda, DEVICE_ID, Al>::randint_like(self.inner.as_ref(), low, high)?.into())
    }
}
