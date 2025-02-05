use std::{cell::RefCell, rc::Rc};

use crate::{
    tensor::{DiffTensor, Tensor},
    tensor_base::_Tensor,
    Cpu,
};
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Open01, OpenClosed01, Standard, StandardNormal,
};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{CommonBounds, Random, RandomInt};
use hpt_types::into_scalar::Cast;

impl<T, const DEVICE: usize> Random for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + SampleUniform + num::Float + rand_distr::num_traits::FloatConst,
    <T as SampleUniform>::Sampler: Sync,
    StandardNormal: Distribution<T>,
    Open01: Distribution<T>,
    Exp1: Distribution<T>,
    OpenClosed01: Distribution<T>,
    Standard: Distribution<T>,
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::randn(shape)?.into())
    }

    fn randn_like(&self) -> Result<Self, TensorError> {
        Ok(_Tensor::randn_like(self.inner.as_ref())?.into())
    }

    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::rand(shape, low, high)?.into())
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::rand_like(self.inner.as_ref(), low, high)?.into())
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::beta(a, b, shape)?.into())
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::beta_like(self.inner.as_ref(), a, b)?.into())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::chisquare(df, shape)?.into())
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::chisquare_like(self.inner.as_ref(), df)?.into())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::exponential(lambda, shape)?.into())
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::exponential_like(self.inner.as_ref(), lambda)?.into())
    }

    fn gamma<S: Into<Shape>>(
        gamm_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::gamma(gamm_shape, scale, shape)?.into())
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::gamma_like(self.inner.as_ref(), shape, scale)?.into())
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::gumbel(mu, beta, shape)?.into())
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::gumbel_like(self.inner.as_ref(), mu, beta)?.into())
    }

    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::lognormal(mean, std, shape)?.into())
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::lognormal_like(self.inner.as_ref(), mean, std)?.into())
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::normal_gaussian(mean, std, shape)?.into())
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::normal_gaussian_like(self.inner.as_ref(), mean, std)?.into())
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::pareto(pareto_shape, a, shape)?.into())
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::pareto_like(self.inner.as_ref(), pareto_shape, a)?.into())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::poisson(lambda, shape)?.into())
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::poisson_like(self.inner.as_ref(), lambda)?.into())
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::weibull(a, b, shape)?.into())
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::weibull_like(self.inner.as_ref(), a, b)?.into())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(_Tensor::zipf(n, a, shape)?.into())
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(_Tensor::zipf_like(self.inner.as_ref(), n, a)?.into())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::triangular(low, high, mode, shape)?.into())
    }

    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(_Tensor::triangular_like(self.inner.as_ref(), low, high, mode)?.into())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        Ok(_Tensor::bernoulli(shape, p)?.into())
    }
}

impl<T, const DEVICE: usize> RandomInt for Tensor<T, Cpu, DEVICE>
where
    T: CommonBounds + SampleUniform,
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
        Ok(_Tensor::randint(low, high, shape)?.into())
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(_Tensor::randint_like(self.inner.as_ref(), low, high)?.into())
    }
}

impl<T, const DEVICE: usize> Random for DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds + SampleUniform + num::Float + rand_distr::num_traits::FloatConst,
    <T as SampleUniform>::Sampler: Sync,
    StandardNormal: Distribution<T>,
    Open01: Distribution<T>,
    Exp1: Distribution<T>,
    OpenClosed01: Distribution<T>,
    Standard: Distribution<T>,
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::randn(shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn randn_like(&self) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.randn_like()?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::rand(shape, low, high)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.rand_like(low, high)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::beta(a, b, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.beta_like(a, b)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::chisquare(df, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.chisquare_like(df)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::exponential(lambda, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.exponential_like(lambda)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gamma<S: Into<Shape>>(
        gamm_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::gamma(gamm_shape, scale, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.gamma_like(shape, scale)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::gumbel(mu, beta, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.gumbel_like(mu, beta)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::lognormal(mean, std, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.lognormal_like(mean, std)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::normal_gaussian(mean, std, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.normal_gaussian_like(mean, std)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::pareto(pareto_shape, a, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.pareto_like(pareto_shape, a)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::poisson(lambda, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.poisson_like(lambda)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::weibull(a, b, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.weibull_like(a, b)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::zipf(n, a, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.zipf_like(n, a)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: Tensor::triangular(low, high, mode, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError> {
        Ok(DiffTensor {
            inner: self.inner.triangular_like(low, high, mode)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        Ok(DiffTensor {
            inner: Tensor::bernoulli(shape, p)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }
}

impl<T, const DEVICE: usize> RandomInt for DiffTensor<T, Cpu, DEVICE>
where
    T: CommonBounds + SampleUniform,
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
        Ok(DiffTensor {
            inner: Tensor::randint(low, high, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Ok(DiffTensor {
            inner: self.inner.randint_like(low, high)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }
}
