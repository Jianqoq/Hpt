use std::{cell::RefCell, rc::Rc};

use crate::{
    tensor::{DiffTensor, Tensor},
    Cpu,
};
use anyhow::Result;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Open01, OpenClosed01, Standard, StandardNormal,
};
use tensor_common::shape::shape::Shape;
use tensor_traits::{CommonBounds, Random, RandomInt};
use tensor_types::into_scalar::IntoScalar;

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

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::randn(shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn randn_like(&self) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.randn_like()?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::rand(shape, low, high)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn rand_like(&self) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.rand_like()?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::beta(a, b, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.beta_like(a, b)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::chisquare(df, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.chisquare_like(df)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::exponential(lambda, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.exponential_like(lambda)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gamma<S: Into<Shape>>(gamm_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::gamma(gamm_shape, scale, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.gamma_like(shape, scale)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::gumbel(mu, beta, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.gumbel_like(mu, beta)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::lognormal(mean, std, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
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
    ) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::normal_gaussian(mean, std, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.normal_gaussian_like(mean, std)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::pareto(pareto_shape, a, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.pareto_like(pareto_shape, a)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::poisson(lambda, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.poisson_like(lambda)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::weibull(a, b, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.weibull_like(a, b)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::zipf(n, a, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
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
    ) -> Result<Self> {
        Ok(DiffTensor {
            inner: Tensor::triangular(low, high, mode, shape)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        Ok(DiffTensor {
            inner: self.inner.triangular_like(low, high, mode)?,
            grad: Rc::new(RefCell::new(None)),
            out_degree: Rc::new(RefCell::new(0)),
            backward: Rc::new(RefCell::new(move |_| Ok(true))),
        })
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
    where
        T: IntoScalar<f64>,
        bool: IntoScalar<T>,
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

    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
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

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
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
