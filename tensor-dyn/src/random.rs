use rand_distr::{
    uniform::SampleUniform,
    Distribution,
    Exp1,
    Normal,
    NormalInverseGaussian,
    Open01,
    OpenClosed01,
    Standard,
    StandardNormal,
    Uniform,
};
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use tensor_common::shape::Shape;
use tensor_traits::{ random::Random, tensor::{ CommonBounds, TensorCreator, TensorInfo } };
use anyhow::Result;
use crate::tensor_base::_Tensor;

impl<T> Random
    for _Tensor<T>
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
        let res_shape = Shape::from(shape.into());
        let ret: _Tensor<T> = _Tensor::empty(res_shape)?;
        let normal: Normal<T> = Normal::new(T::from(0.0).unwrap(), T::from(1.0).unwrap())?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn randn_like(&self) -> Result<Self> {
        return _Tensor::randn(self.shape());
    }

    fn rand<S: Into<Shape>>(shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: Uniform<T> = Uniform::new(T::from(0.0).unwrap(), T::from(1.0).unwrap());
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn rand_like(&self) -> Result<Self> {
        return _Tensor::randn(self.shape().clone());
    }

    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: Uniform<T> = Uniform::new(low, high);
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self> {
        return _Tensor::randint(low, high, self.shape().clone());
    }

    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::Beta<T> = rand_distr::Beta::new(a, b)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        return _Tensor::beta(a, b, self.shape().clone());
    }

    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::ChiSquared<T> = rand_distr::ChiSquared::new(df)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn chisquare_like(&self, df: Self::Meta) -> Result<Self> {
        return _Tensor::chisquare(df, self.shape().clone());
    }

    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::Exp<T> = rand_distr::Exp::new(lambda)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self> {
        return _Tensor::exponential(lambda, self.shape().clone());
    }

    fn gamma<S: Into<Shape>>(gamma_shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::Gamma<T> = rand_distr::Gamma::new(gamma_shape, scale)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn gamma_like(&self, gamma_shape: Self::Meta, scale: Self::Meta) -> Result<Self> {
        return _Tensor::gamma(gamma_shape, scale, self.shape().clone());
    }

    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::Gumbel<T> = rand_distr::Gumbel::new(mu, beta)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self> {
        return _Tensor::gumbel(mu, beta, self.shape().clone());
    }

    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: rand_distr::LogNormal<T> = rand_distr::LogNormal::new(mean, std)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    let rand_num = normal.sample(rng);
                    *x = rand_num;
                }
            );
        return Ok(ret);
    }

    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        return _Tensor::lognormal(mean, std, self.shape().clone());
    }

    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S
    ) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let normal: NormalInverseGaussian<T> = NormalInverseGaussian::new(mean, std)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = normal.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self> {
        return _Tensor::normal_gaussian(mean, std, self.shape().clone());
    }

    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let pareto: rand_distr::Pareto<T> = rand_distr::Pareto::new(a, pareto_shape)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = pareto.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self> {
        return _Tensor::pareto(pareto_shape, a, self.shape().clone());
    }

    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let poisson: rand_distr::Poisson<T> = rand_distr::Poisson::new(lambda)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = poisson.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self> {
        return _Tensor::poisson(lambda, self.shape().clone());
    }

    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let weibull: rand_distr::Weibull<T> = rand_distr::Weibull::new(a, b)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = weibull.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self> {
        return _Tensor::weibull(a, b, self.shape().clone());
    }

    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let zipf: rand_distr::Zipf<T> = rand_distr::Zipf::new(n, a)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = zipf.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self> {
        return _Tensor::zipf(n, a, self.shape().clone());
    }

    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S
    ) -> Result<Self> {
        let res_shape = Shape::from(shape.into());
        let ret = _Tensor::empty(res_shape)?;
        let triangular: rand_distr::Triangular<T> = rand_distr::Triangular::new(low, high, mode)?;
        ret.as_raw_mut()
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, x| {
                    *x = triangular.sample(rng);
                }
            );
        return Ok(ret);
    }

    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self> {
        return _Tensor::triangular(low, high, mode, self.shape().clone());
    }
}
