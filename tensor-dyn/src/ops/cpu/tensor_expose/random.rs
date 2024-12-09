use crate::{ tensor::Tensor, tensor_base::_Tensor };
use anyhow::Result;
use rand_distr::{
    uniform::SampleUniform,
    Distribution,
    Exp1,
    Open01,
    OpenClosed01,
    Standard,
    StandardNormal,
};
use tensor_common::shape::Shape;
use tensor_traits::{ CommonBounds, Random, RandomInt };
use tensor_types::into_scalar::IntoScalar;

impl<T> Random
    for Tensor<T>
    where
        T: CommonBounds + SampleUniform + num::Float + rand_distr::num_traits::FloatConst,
        <T as SampleUniform>::Sampler: Sync,
        StandardNormal: Distribution<T>,
        Open01: Distribution<T>,
        Exp1: Distribution<T>,
        OpenClosed01: Distribution<T>,
        Standard: Distribution<T>
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::randn(shape)?.into())
    }

    fn randn_like(&self) -> Result<Self> {
        Ok(_Tensor::<T>::randn_like(self.inner.as_ref())?.into())
    }

    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::rand(shape, low, high)?.into())
    }

    fn rand_like(&self) -> Result<Self> {
        Ok(_Tensor::<T>::rand_like(self.inner.as_ref())?.into())
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::beta(a, b, shape)?.into())
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::beta_like(self.inner.as_ref(), a, b)?.into())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::chisquare(df, shape)?.into())
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::chisquare_like(self.inner.as_ref(), df)?.into())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::exponential(lambda, shape)?.into())
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::exponential_like(self.inner.as_ref(), lambda)?.into())
    }

    fn gamma<S: Into<Shape>>(gamm_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::gamma(gamm_shape, scale, shape)?.into())
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::gamma_like(self.inner.as_ref(), shape, scale)?.into())
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::gumbel(mu, beta, shape)?.into())
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::gumbel_like(self.inner.as_ref(), mu, beta)?.into())
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::lognormal(mean, std, shape)?.into())
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::lognormal_like(self.inner.as_ref(), mean, std)?.into())
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S
    ) -> Result<Self> {
        Ok(_Tensor::<T>::normal_gaussian(mean, std, shape)?.into())
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::normal_gaussian_like(self.inner.as_ref(), mean, std)?.into())
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::pareto(pareto_shape, a, shape)?.into())
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::pareto_like(self.inner.as_ref(), pareto_shape, a)?.into())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::poisson(lambda, shape)?.into())
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::poisson_like(self.inner.as_ref(), lambda)?.into())
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::weibull(a, b, shape)?.into())
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::weibull_like(self.inner.as_ref(), a, b)?.into())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(_Tensor::<T>::zipf(n, a, shape)?.into())
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::zipf_like(self.inner.as_ref(), n, a)?.into())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S
    ) -> Result<Self> {
        Ok(_Tensor::<T>::triangular(low, high, mode, shape)?.into())
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        Ok(_Tensor::<T>::triangular_like(self.inner.as_ref(), low, high, mode)?.into())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
        where T: IntoScalar<f64>, bool: IntoScalar<T>
    {
        Ok(_Tensor::<T>::bernoulli(shape, p)?.into())
    }
}

impl<T> RandomInt for Tensor<T> where T: CommonBounds + SampleUniform {
    type Meta = T;

    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
        where <T as SampleUniform>::Sampler: Sync
    {
        Ok(_Tensor::<T>::randint(low, high, shape)?.into())
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
        where <T as SampleUniform>::Sampler: Sync
    {
        Ok(_Tensor::<T>::randint_like(self.inner.as_ref(), low, high)?.into())
    }
}
