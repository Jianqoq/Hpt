#![allow(unused)]

use crate::{tensor_base::_Tensor, Cpu, Cuda, CUDA_SEED};
use cudarc::driver::DeviceRepr;
use hpt_allocator::traits::{Allocator, AllocatorOutputRetrive};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{
    random::Random,
    tensor::{CommonBounds, TensorCreator, TensorInfo},
    RandomInt, TensorLike,
};
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Normal, NormalInverseGaussian, Open01,
    OpenClosed01, Standard, StandardNormal, Uniform,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl<T, const DEVICE_ID: usize, Al> Random for _Tensor<T, Cuda, DEVICE_ID, Al>
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
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID, Al>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            CUDA_SEED.load(std::sync::atomic::Ordering::Relaxed),
            ret.device(),
        )
        .expect("CUDA_RNG error");
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_normal(&mut cuda_slice, T::ZERO, T::ONE)
            .expect("CUDA_RNG error");
        cuda_slice.leak();
        Ok(ret)
    }

    fn randn_like(&self) -> Result<Self, TensorError> {
        _Tensor::randn(self.shape())
    }

    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID, Al>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            CUDA_SEED.load(std::sync::atomic::Ordering::Relaxed),
            ret.device(),
        )
        .expect("CUDA_RNG error");
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_uniform(&mut cuda_slice)
            .expect("CUDA_RNG error");
        cuda_slice.leak();
        Ok(ret)
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::rand(self.shape().clone(), low, high)
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::beta(a, b, self.shape().clone())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::chisquare(df, self.shape().clone())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::exponential(lambda, self.shape().clone())
    }

    fn gamma<S: Into<Shape>>(
        gamma_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn gamma_like(&self, gamma_shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::gamma(gamma_shape, scale, self.shape().clone())
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::gumbel(mu, beta, self.shape().clone())
    }

    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::<T, Cuda, DEVICE_ID, Al>::empty(res_shape)?;
        let rng = cudarc::curand::CudaRng::new(
            CUDA_SEED.load(std::sync::atomic::Ordering::Relaxed),
            ret.device(),
        )
        .expect("CUDA_RNG error");
        let mut cuda_slice = unsafe {
            ret.device()
                .upgrade_device_ptr::<T>(ret.ptr().ptr as u64, ret.size())
        };
        rng.fill_with_log_normal(&mut cuda_slice, mean, std)
            .expect("CUDA_RNG error");
        cuda_slice.leak();
        Ok(ret)
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::lognormal(mean, std, self.shape().clone())
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::normal_gaussian(mean, std, self.shape().clone())
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::pareto(pareto_shape, a, self.shape().clone())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::poisson(lambda, self.shape().clone())
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::weibull(a, b, self.shape().clone())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::zipf(n, a, self.shape().clone())
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        unimplemented!()
    }

    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError> {
        _Tensor::triangular(low, high, mode, self.shape().clone())
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        unimplemented!()
    }
}

impl<T, const DEVICE_ID: usize, Al> RandomInt for _Tensor<T, Cuda, DEVICE_ID, Al>
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
        unimplemented!()
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        _Tensor::<T, Cuda, DEVICE_ID, Al>::randint(low, high, self.shape().clone())
    }
}
