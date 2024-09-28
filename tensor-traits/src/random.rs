use anyhow::Result;
use rand_distr::uniform::SampleUniform;
use tensor_common::shape::Shape;
use tensor_types::into_scalar::IntoScalar;

/// A trait for generating random numbers.
pub trait Random where Self: Sized {
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// Generates a random number array with a standard normal distribution (mean = `0`, standard deviation = `1`).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randn<S: Into<Shape>>(shape: S) -> Result<Self>;

    /// Generates a random number array with a standard normal distribution (mean = `0`, standard deviation = `1`),
    /// with the same shape as the calling instance.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randn_like(&self) -> Result<Self>;

    /// Generates a random number array with a uniform distribution between [0, 1).
    ///
    /// # Parameters
    /// - `low`: The lower bound of the uniform distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self>;

    /// Generates a random number array with a uniform distribution between [0, 1),
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rand_like(&self) -> Result<Self>;

    /// Generates a random number array following the Beta distribution.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the alpha parameter of the Beta distribution.
    /// - `b`: A value of type `Self::Meta` representing the beta parameter of the Beta distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Beta distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the alpha parameter of the Beta distribution.
    /// - `b`: A value of type `Self::Meta` representing the beta parameter of the Beta distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Chi-Squared distribution.
    ///
    /// # Parameters
    /// - `df`: A value of type `Self::Meta` representing the degrees of freedom of the Chi-Squared distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Chi-Squared distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `df`: A value of type `Self::Meta` representing the degrees of freedom of the Chi-Squared distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn chisquare_like(&self, df: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Exponential distribution.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter of the Exponential distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Exponential distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter of the Exponential distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Gamma distribution.
    ///
    /// # Parameters
    /// - `shape`: A value of type `Self::Meta` representing the shape parameter of the Gamma distribution.
    /// - `scale`: A value of type `Self::Meta` representing the scale parameter of the Gamma distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gamma<S: Into<Shape>>(shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Gamma distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `shape`: A value of type `Self::Meta` representing the shape parameter of the Gamma distribution.
    /// - `scale`: A value of type `Self::Meta` representing the scale parameter of the Gamma distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Gumbel distribution.
    ///
    /// # Parameters
    /// - `mu`: A value of type `Self::Meta` representing the location parameter of the Gumbel distribution.
    /// - `beta`: A value of type `Self::Meta` representing the scale parameter of the Gumbel distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Gumbel distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mu`: A value of type `Self::Meta` representing the location parameter of the Gumbel distribution.
    /// - `beta`: A value of type `Self::Meta` representing the scale parameter of the Gumbel distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Log-Normal distribution.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the underlying normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the underlying normal distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Log-Normal distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the underlying normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the underlying normal distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Normal (Gaussian) distribution.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the Normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the Normal distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn normal_gaussian<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Normal (Gaussian) distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the Normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the Normal distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Pareto distribution.
    ///
    /// # Parameters
    /// - `pareto_shape`: A value of type `Self::Meta` representing the shape parameter of the Pareto distribution.
    /// - `a`: A value of type `Self::Meta` representing the scale parameter 'a' of the Pareto distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Pareto distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `pareto_shape`: A value of type `Self::Meta` representing the shape parameter of the Pareto distribution.
    /// - `a`: A value of type `Self::Meta` representing the scale parameter 'a' of the Pareto distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Poisson distribution.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter (λ) of the Poisson distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Poisson distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter (λ) of the Poisson distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Weibull distribution.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the shape parameter of the Weibull distribution.
    /// - `b`: A value of type `Self::Meta` representing the scale parameter of the Weibull distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Weibull distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the shape parameter of the Weibull distribution.
    /// - `b`: A value of type `Self::Meta` representing the scale parameter of the Weibull distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Zipf distribution.
    ///
    /// # Parameters
    /// - `n`: A `u64` value representing the number of elements (size) of the Zipf distribution.
    /// - `a`: A value of type `Self::Meta` representing the exponent parameter of the Zipf distribution.
    /// - `shape`: The shape of the output array, can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Zipf distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `n`: A `u64` value representing the number of elements (size) of the Zipf distribution.
    /// - `a`: A value of type `Self::Meta` representing the exponent parameter of the Zipf distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Triangular distribution.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower limit of the distribution.
    /// - `high`: A value of type `Self::Meta` representing the upper limit of the distribution.
    /// - `mode`: A value of type `Self::Meta` representing the mode (peak) of the distribution.
    /// - `shape`: The shape of the output array, can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S
    ) -> Result<Self>;

    /// Generates a random number array following the Triangular distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower limit of the distribution.
    /// - `high`: A value of type `Self::Meta` representing the upper limit of the distribution.
    /// - `mode`: A value of type `Self::Meta` representing the mode (peak) of the distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self>;

    /// Generates a bernoulli array with values `true` or `false` following the Bernoulli distribution.
    ///
    /// # Parameters
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    /// - `p`: A value of type `Self::Meta` representing the probability of success (true) in the Bernoulli distribution.
    /// # Returns
    /// A random boolean array with values `true` or `false` following the Bernoulli distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
        where Self::Meta: IntoScalar<f64>, bool: IntoScalar<Self::Meta>;
}

/// A trait for generating random integers.
pub trait RandomInt where Self: Sized {
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// Generates a random integer array with values in the specified range.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower bound of the range (inclusive).
    /// - `high`: A value of type `Self::Meta` representing the upper bound of the range (exclusive).
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
        where Self::Meta: SampleUniform, <Self::Meta as SampleUniform>::Sampler: Sync;

    /// Generates a random integer array with values in the specified range,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower bound of the range (inclusive).
    /// - `high`: A value of type `Self::Meta` representing the upper bound of the range (exclusive).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
        where Self::Meta: SampleUniform, <Self::Meta as SampleUniform>::Sampler: Sync;
}
