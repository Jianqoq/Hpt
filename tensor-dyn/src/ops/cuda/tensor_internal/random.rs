#![allow(unused)]

use crate::{backend::Cpu, tensor_base::_Tensor, Cuda};
use anyhow::Result;
use cudarc::{driver::DeviceRepr, types::CudaTypeName};
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Normal, NormalInverseGaussian, Open01,
    OpenClosed01, Standard, StandardNormal, Uniform,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tensor_common::shape::Shape;
use tensor_traits::{
    random::Random,
    tensor::{CommonBounds, TensorCreator, TensorInfo},
    RandomInt, TensorLike,
};
use tensor_types::into_scalar::Cast;

impl<T, const DEVICE_ID: usize> Random for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds
        + SampleUniform
        + num::Float
        + rand_distr::num_traits::FloatConst
        + DeviceRepr
        + CudaTypeName,
    <T as SampleUniform>::Sampler: Sync,
    StandardNormal: Distribution<T>,
    Open01: Distribution<T>,
    Exp1: Distribution<T>,
    OpenClosed01: Distribution<T>,
    Standard: Distribution<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::NormalFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::UniformFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::LogNormalFill<T>,
{
    type Meta = T;
    fn randn<S: Into<Shape>>(shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ret.device(),
        )?;
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_normal(&mut cuda_slice, T::ZERO, T::ONE)?;
        Ok(ret)
    }

    fn randn_like(&self) -> Result<Self> {
        _Tensor::randn(self.shape())
    }

    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ret.device(),
        )?;
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_uniform(&mut cuda_slice)?;
        Ok(ret)
    }

    fn rand_like(&self) -> Result<Self> {
        _Tensor::randn(self.shape().clone())
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        _Tensor::beta(a, b, self.shape().clone())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        _Tensor::chisquare(df, self.shape().clone())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        _Tensor::exponential(lambda, self.shape().clone())
    }

    fn gamma<S: Into<Shape>>(gamma_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn gamma_like(&self, gamma_shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        _Tensor::gamma(gamma_shape, scale, self.shape().clone())
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        _Tensor::gumbel(mu, beta, self.shape().clone())
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ret.device(),
        )?;
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_log_normal(&mut cuda_slice, mean, std)?;
        Ok(ret)
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        _Tensor::lognormal(mean, std, self.shape().clone())
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self> {
        unimplemented!()
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        _Tensor::normal_gaussian(mean, std, self.shape().clone())
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        _Tensor::pareto(pareto_shape, a, self.shape().clone())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        _Tensor::poisson(lambda, self.shape().clone())
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        _Tensor::weibull(a, b, self.shape().clone())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        unimplemented!()
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
        _Tensor::zipf(n, a, self.shape().clone())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self> {
        unimplemented!()
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        _Tensor::triangular(low, high, mode, self.shape().clone())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        unimplemented!()
    }
}

impl<T, const DEVICE_ID: usize> RandomInt for _Tensor<T, Cuda, DEVICE_ID>
where
    T: CommonBounds + SampleUniform,
{
    type Meta = T;
    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        unimplemented!()
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        _Tensor::<T, Cuda, DEVICE_ID>::randint(low, high, self.shape().clone())
    }
}
