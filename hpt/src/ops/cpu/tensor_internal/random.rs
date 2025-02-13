use crate::{backend::Cpu, tensor_base::_Tensor};
use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_traits::{
    random::Random,
    tensor::{CommonBounds, TensorCreator, TensorInfo},
    RandomInt, TensorLike,
};
use hpt_types::into_scalar::Cast;
use rand_distr::{
    uniform::SampleUniform, Distribution, Exp1, Normal, NormalInverseGaussian, Open01,
    OpenClosed01, Standard, StandardNormal, Uniform,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl<T, const DEVICE: usize> Random for _Tensor<T, Cpu, DEVICE>
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
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = Normal::new(T::from(0.0).unwrap(), T::from(1.0).unwrap())?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
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
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = Uniform::new(low, high);
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::rand(self.shape().clone(), low, high)
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::Beta::new(a, b)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::beta(a, b, self.shape().clone())
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::ChiSquared::new(df)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::chisquare(df, self.shape().clone())
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::Exp::new(lambda)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::exponential(lambda, self.shape().clone())
    }

    fn gamma<S: Into<Shape>>(
        gamma_shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::Gamma::new(gamma_shape, scale)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn gamma_like(&self, gamma_shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::gamma(gamma_shape, scale, self.shape().clone())
    }

    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::Gumbel::new(mu, beta)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
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
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = rand_distr::LogNormal::new(mean, std)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
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
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = NormalInverseGaussian::new(mean, std)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = normal.sample(rng);
            },
        );
        Ok(ret)
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::normal_gaussian(mean, std, self.shape().clone())
    }

    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let pareto = rand_distr::Pareto::new(a, pareto_shape)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = pareto.sample(rng);
            },
        );
        Ok(ret)
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::pareto(pareto_shape, a, self.shape().clone())
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let poisson = rand_distr::Poisson::new(lambda)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = poisson.sample(rng);
            },
        );
        Ok(ret)
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::poisson(lambda, self.shape().clone())
    }

    fn weibull<S: Into<Shape>>(
        a: Self::Meta,
        b: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let weibull = rand_distr::Weibull::new(a, b)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = weibull.sample(rng);
            },
        );
        Ok(ret)
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError> {
        _Tensor::weibull(a, b, self.shape().clone())
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self, TensorError> {
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let zipf = rand_distr::Zipf::new(n, a)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = zipf.sample(rng);
            },
        );
        Ok(ret)
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
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let triangular = rand_distr::Triangular::new(low, high, mode)?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = triangular.sample(rng);
            },
        );
        Ok(ret)
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
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let bernoulli = rand_distr::Bernoulli::new(p.cast())?;
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                *x = bernoulli.sample(rng).cast();
            },
        );
        Ok(ret)
    }
}

impl<T, const DEVICE: usize> RandomInt for _Tensor<T, Cpu, DEVICE>
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
        let res_shape = Shape::from(shape.into());
        let mut ret = _Tensor::empty(res_shape)?;
        let normal = Uniform::new(low, high);
        ret.as_raw_mut().into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rng, x| {
                let rand_num = normal.sample(rng);
                *x = rand_num;
            },
        );
        Ok(ret)
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        <T as SampleUniform>::Sampler: Sync,
    {
        _Tensor::randint(low, high, self.shape().clone())
    }
}
