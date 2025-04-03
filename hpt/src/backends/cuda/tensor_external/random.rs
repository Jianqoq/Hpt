use crate::{backend::Cuda, tensor::Tensor};
use cudarc::driver::DeviceRepr;
use hpt_allocator::{
    traits::{Allocator, AllocatorOutputRetrive},
    Cpu,
};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{
    ops::random::{Random, RandomInt},
    tensor::{CommonBounds, TensorInfo},
};
use hpt_types::dtype::CudaType;
use hpt_types::into_scalar::Cast;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Open01, OpenClosed01, StandardNormal,
    StandardUniform,
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
    StandardUniform: Distribution<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::NormalFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::UniformFill<T>,
    cudarc::curand::sys::curandGenerator_t: cudarc::curand::result::LogNormalFill<T>,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    Al::CpuAllocator: Allocator<CudaAllocator = Al>,
{
    type Meta = T;

    fn randn<S: Into<Shape>>(shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::randn(shape)?.to_cuda::<DEVICE_ID>()
    }

    fn randn_like(&self) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::randn(self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::rand(shape, low, high)?
            .to_cuda::<DEVICE_ID>()
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::rand(self.shape(), low, high)?
            .to_cuda::<DEVICE_ID>()
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::beta(a, b, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::beta(a, b, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::chisquare(df, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::chisquare(df, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::exponential(lambda, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::exponential(lambda, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn gamma<S: Into<Shape>>(
        gamm_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::gamma(gamm_shape, scale, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::gamma(shape, scale, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::gumbel(mu, beta, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::gumbel(mu, beta, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::lognormal(mean, std, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::lognormal(mean, std, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::normal_gaussian(mean, std, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::normal_gaussian(
            mean,
            std,
            self.shape(),
        )?
        .to_cuda::<DEVICE_ID>()
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::pareto(pareto_shape, a, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::pareto(pareto_shape, a, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::poisson(lambda, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::poisson(lambda, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::weibull(a, b, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::weibull(a, b, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn zipf<S: Into<Shape>>(n: Self::Meta, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::zipf(n, a, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn zipf_like(&self, n: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::zipf(n, a, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::triangular(low, high, mode, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError> {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::triangular(
            low,
            high,
            mode,
            self.shape(),
        )?
        .to_cuda::<DEVICE_ID>()
    }

    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        T: Cast<f64>,
        bool: Cast<T>,
    {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::bernoulli(shape, p)?
            .to_cuda::<DEVICE_ID>()
    }
}

impl<T, const DEVICE_ID: usize, Al> RandomInt for Tensor<T, Cuda, DEVICE_ID, Al>
where
    T: CommonBounds + SampleUniform + DeviceRepr + CudaType,
    Al: Allocator,
    Al::Output: AllocatorOutputRetrive,
    Al::CpuAllocator: Allocator<CudaAllocator = Al>,
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
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::randint(low, high, shape)?
            .to_cuda::<DEVICE_ID>()
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        Tensor::<T, Cpu, 0, <Al as Allocator>::CpuAllocator>::randint(low, high, self.shape())?
            .to_cuda::<DEVICE_ID>()
    }
}
