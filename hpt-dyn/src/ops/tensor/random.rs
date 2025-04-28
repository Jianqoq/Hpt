use hpt_common::error::base::TensorError;
use hpt_traits::tensor::TensorInfo;
use hpt_types::{dtype::DType, into_scalar::Cast};
use rand_distr::{Distribution, Normal, Uniform};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{Device, Tensor};
use half::{bf16, f16};

pub(crate) fn random_template<F32Func, F16Func, BF16Func>(
    random_method: &str,
    shape: &[i64],
    dtype: DType,
    device: Device,
    f32_func: F32Func,
    f16_func: F16Func,
    bf16_func: BF16Func,
) -> Result<Tensor, TensorError>
where
    F32Func: Fn(&mut [f32]),
    F16Func: Fn(&mut [f16]),
    BF16Func: Fn(&mut [bf16]),
{
    let mut ret = Tensor::empty(shape, dtype, device)?;
    match dtype {
        DType::Bool => panic!("Bool type does not support {random_method}"),
        DType::I8 => panic!("I8 type does not support {random_method}"),
        DType::U8 => panic!("U8 type does not support {random_method}"),
        DType::I16 => panic!("I16 type does not support {random_method}"),
        DType::U16 => panic!("U16 type does not support {random_method}"),
        DType::I32 => panic!("I32 type does not support {random_method}"),
        DType::U32 => panic!("U32 type does not support {random_method}"),
        DType::I64 => panic!("I64 type does not support {random_method}"),
        DType::F32 => f32_func(&mut ret.as_slice_mut::<f32>()),
        DType::F16 => f16_func(&mut ret.as_slice_mut::<f16>()),
        DType::BF16 => bf16_func(&mut ret.as_slice_mut::<bf16>()),
    }
    Ok(ret)
}

impl Tensor {
    pub fn randn(
        mean: f64,
        std: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "randn",
            shape,
            dtype,
            device,
            |slice| {
                let normal = Normal::new(mean as f32, std as f32)
                    .expect("Failed to create normal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = Normal::new(f16::from_f64(mean), f16::from_f64(std))
                    .expect("Failed to create normal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = Normal::new(bf16::from_f64(mean), bf16::from_f64(std))
                    .expect("Failed to create normal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn randn_like(&self) -> Result<Self, TensorError> {
        Self::randn(0.0, 1.0, self.shape(), self.dtype, self.device.clone())
    }

    pub fn rand(
        low: f64,
        high: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "rand",
            shape,
            dtype,
            device,
            |slice| {
                let normal = Uniform::new(low as f32, high as f32)
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = Uniform::new(f16::from_f64(low), f16::from_f64(high))
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = Uniform::new(bf16::from_f64(low), bf16::from_f64(high))
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn rand_like(&self, low: f64, high: f64) -> Result<Self, TensorError> {
        Self::rand(low, high, self.shape(), self.dtype, self.device.clone())
    }

    pub fn beta(
        alpha: f64,
        beta: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "beta",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Beta::new(alpha as f32, beta as f32)
                    .expect("Failed to create beta distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Beta::new(f16::from_f64(alpha), f16::from_f64(beta))
                    .expect("Failed to create beta distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Beta::new(bf16::from_f64(alpha), bf16::from_f64(beta))
                    .expect("Failed to create beta distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn beta_like(&self, alpha: f64, beta: f64) -> Result<Self, TensorError> {
        Self::beta(alpha, beta, self.shape(), self.dtype, self.device.clone())
    }

    pub fn chisquare(
        k: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "chisquare",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::ChiSquared::new(k as f32)
                    .expect("Failed to create chi-square distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::ChiSquared::new(f16::from_f64(k))
                    .expect("Failed to create chi-square distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::ChiSquared::new(bf16::from_f64(k))
                    .expect("Failed to create chi-square distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn chisquare_like(&self, df: f64) -> Result<Self, TensorError> {
        Self::chisquare(df, self.shape(), self.dtype, self.device.clone())
    }

    pub fn exponential(
        lambda: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "exponential",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Exp::new(lambda as f32)
                    .expect("Failed to create exponential distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Exp::new(f16::from_f64(lambda))
                    .expect("Failed to create exponential distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Exp::new(bf16::from_f64(lambda))
                    .expect("Failed to create exponential distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn exponential_like(&self, lambda: f64) -> Result<Self, TensorError> {
        Self::exponential(lambda, self.shape(), self.dtype, self.device.clone())
    }

    pub fn gamma(
        gamma_shape: f64,
        scale: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "gamma",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Gamma::new(gamma_shape as f32, scale as f32)
                    .expect("Failed to create gamma distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Gamma::new(f16::from_f64(gamma_shape), f16::from_f64(scale))
                        .expect("Failed to create gamma distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Gamma::new(bf16::from_f64(gamma_shape), bf16::from_f64(scale))
                        .expect("Failed to create gamma distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn gamma_like(&self, gamma_shape: f64, scale: f64) -> Result<Self, TensorError> {
        Self::gamma(
            gamma_shape,
            scale,
            self.shape(),
            self.dtype,
            self.device.clone(),
        )
    }

    pub fn gumbel(
        location: f64,
        scale: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "gumbel",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Gumbel::new(location as f32, scale as f32)
                    .expect("Failed to create gumbel distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Gumbel::new(f16::from_f64(location), f16::from_f64(scale))
                    .expect("Failed to create gumbel distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Gumbel::new(bf16::from_f64(location), bf16::from_f64(scale))
                    .expect("Failed to create gumbel distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn gumbel_like(&self, mu: f64, beta: f64) -> Result<Self, TensorError> {
        Self::gumbel(mu, beta, self.shape(), self.dtype, self.device.clone())
    }

    pub fn lognormal(
        mu: f64,
        sigma: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "lognormal",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::LogNormal::new(mu as f32, sigma as f32)
                    .expect("Failed to create lognormal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::LogNormal::new(f16::from_f64(mu), f16::from_f64(sigma))
                    .expect("Failed to create lognormal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::LogNormal::new(bf16::from_f64(mu), bf16::from_f64(sigma))
                    .expect("Failed to create lognormal distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn lognormal_like(&self, mu: f64, sigma: f64) -> Result<Self, TensorError> {
        Self::lognormal(mu, sigma, self.shape(), self.dtype, self.device.clone())
    }

    pub fn normal_inverse_gaussian(
        alpha: f64,
        beta: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "normal_inverse_gaussian",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::NormalInverseGaussian::new(alpha as f32, beta as f32)
                    .expect("Failed to create normal inverse gaussian distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::NormalInverseGaussian::new(
                    f16::from_f64(alpha),
                    f16::from_f64(beta),
                )
                .expect("Failed to create normal inverse gaussian distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::NormalInverseGaussian::new(
                    bf16::from_f64(alpha),
                    bf16::from_f64(beta),
                )
                .expect("Failed to create normal inverse gaussian distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn normal_inverse_gaussian_like(&self, alpha: f64, beta: f64) -> Result<Self, TensorError> {
        Self::normal_inverse_gaussian(alpha, beta, self.shape(), self.dtype, self.device.clone())
    }

    pub fn pareto(
        pareto_shape: f64,
        scale: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "pareto",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Pareto::new(scale as f32, pareto_shape as f32)
                    .expect("Failed to create pareto distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Pareto::new(f16::from_f64(scale), f16::from_f64(pareto_shape))
                        .expect("Failed to create pareto distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Pareto::new(bf16::from_f64(scale), bf16::from_f64(pareto_shape))
                        .expect("Failed to create pareto distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn pareto_like(&self, pareto_shape: f64, scale: f64) -> Result<Self, TensorError> {
        Self::pareto(
            pareto_shape,
            scale,
            self.shape(),
            self.dtype,
            self.device.clone(),
        )
    }

    pub fn poisson(
        lambda: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "poisson",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Poisson::new(lambda as f32)
                    .expect("Failed to create poisson distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Poisson::new(f16::from_f64(lambda))
                    .expect("Failed to create poisson distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Poisson::new(bf16::from_f64(lambda))
                    .expect("Failed to create poisson distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn poisson_like(&self, lambda: f64) -> Result<Self, TensorError> {
        Self::poisson(lambda, self.shape(), self.dtype, self.device.clone())
    }

    pub fn weibull(
        scale: f64,
        weibull_shape: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "weibull",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Weibull::new(scale as f32, weibull_shape as f32)
                    .expect("Failed to create weibull distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Weibull::new(f16::from_f64(scale), f16::from_f64(weibull_shape))
                        .expect("Failed to create weibull distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal =
                    rand_distr::Weibull::new(bf16::from_f64(scale), bf16::from_f64(weibull_shape))
                        .expect("Failed to create weibull distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn weibull_like(&self, scale: f64, weibull_shape: f64) -> Result<Self, TensorError> {
        Self::weibull(
            scale,
            weibull_shape,
            self.shape(),
            self.dtype,
            self.device.clone(),
        )
    }

    pub fn zipf(
        n: f64,
        s: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "zipf",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Zipf::new(n as f32, s as f32)
                    .expect("Failed to create zipf distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Zipf::new(f16::from_f64(n), f16::from_f64(s))
                    .expect("Failed to create zipf distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Zipf::new(bf16::from_f64(n), bf16::from_f64(s))
                    .expect("Failed to create zipf distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn zipf_like(&self, n: f64, s: f64) -> Result<Self, TensorError> {
        Self::zipf(n, s, self.shape(), self.dtype, self.device.clone())
    }

    pub fn triangular(
        low: f64,
        high: f64,
        mode: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "triangular",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Triangular::new(low as f32, high as f32, mode as f32)
                    .expect("Failed to create triangular distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Triangular::new(
                    f16::from_f64(low),
                    f16::from_f64(high),
                    f16::from_f64(mode),
                )
                .expect("Failed to create triangular distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Triangular::new(
                    bf16::from_f64(low),
                    bf16::from_f64(high),
                    bf16::from_f64(mode),
                )
                .expect("Failed to create triangular distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            },
        )
    }

    pub fn triangular_like(&self, low: f64, high: f64, mode: f64) -> Result<Self, TensorError> {
        Self::triangular(
            low,
            high,
            mode,
            self.shape(),
            self.dtype,
            self.device.clone(),
        )
    }

    pub fn bernoulli(
        p: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        random_template(
            "bernoulli",
            shape,
            dtype,
            device,
            |slice| {
                let normal = rand_distr::Bernoulli::new(p as f64)
                    .expect("Failed to create bernoulli distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num.cast();
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Bernoulli::new(p as f64)
                    .expect("Failed to create bernoulli distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num.cast();
                    },
                );
            },
            |slice| {
                let normal = rand_distr::Bernoulli::new(p as f64)
                    .expect("Failed to create bernoulli distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num.cast();
                    },
                );
            },
        )
    }

    pub fn randint(
        low: f64,
        high: f64,
        shape: &[i64],
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let random_method: &str = "randint";
        let mut ret = Tensor::empty(shape, dtype, device)?;
        match dtype {
            DType::Bool => panic!("Bool type does not support {random_method}"),
            DType::I8 => {
                let slice = ret.as_slice_mut::<i8>();
                let normal = Uniform::new(low as i8, high as i8)
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            }
            DType::U8 => panic!("U8 type does not support {random_method}"),
            DType::I16 => {
                let slice = ret.as_slice_mut::<i16>();
                let normal = Uniform::new(low as i16, high as i16)
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            }
            DType::U16 => panic!("U16 type does not support {random_method}"),
            DType::I32 => {
                let slice = ret.as_slice_mut::<i32>();
                let normal = Uniform::new(low as i32, high as i32)
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            }
            DType::U32 => panic!("U32 type does not support {random_method}"),
            DType::I64 => {
                let slice = ret.as_slice_mut::<i64>();
                let normal = Uniform::new(low as i64, high as i64)
                    .expect("Failed to create uniform distribution");
                slice.into_par_iter().for_each_init(
                    || rand::rng(),
                    |rng, x| {
                        let rand_num = normal.sample(rng);
                        *x = rand_num;
                    },
                );
            }
            DType::F32 => panic!("F32 type does not support {random_method}"),
            DType::F16 => panic!("F16 type does not support {random_method}"),
            DType::BF16 => panic!("BF16 type does not support {random_method}"),
        }
        Ok(ret)
    }
}
