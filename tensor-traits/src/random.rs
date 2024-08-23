use anyhow::Result;
use rand_distr::uniform::SampleUniform;
use tensor_common::shape::Shape;
use tensor_types::into_scalar::IntoScalar;

pub trait Random where Self: Sized {
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// Generates a random number array with a standard normal distribution (mean = `0`, standard deviation = `1`).
    ///
    /// # Parameters
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let normal_array = Tensor::<f32>::randn([1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a standard normal distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randn<S: Into<Shape>>(shape: S) -> Result<Self>;

    /// Generates a random number array with a standard normal distribution (mean = `0`, standard deviation = `1`),
    /// with the same shape as the calling instance.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let normal_array_like = a.randn_like()?;
    /// ```
    /// This example creates an array of random numbers following a standard normal distribution,
    /// similar to `a` in shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randn_like(&self) -> Result<Self>;

    /// Generates a random number array with a uniform distribution between [0, 1).
    ///
    /// # Parameters
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let uniform_array = Tensor::<f32>::rand([1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers with a uniform distribution between [0, 1).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rand<S: Into<Shape>>(shape: S, low: Self::Meta, high: Self::Meta) -> Result<Self>;

    /// Generates a random number array with a uniform distribution between [0, 1),
    /// with the same shape as the calling instance.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let uniform_array_like = a.rand_like()?;
    /// ```
    /// This example creates an array of random numbers with a uniform distribution between [0, 1),
    /// similar to `a` in shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn rand_like(&self) -> Result<Self>;

    /// Generates a random number array following the Beta distribution.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the alpha parameter of the Beta distribution.
    /// - `b`: A value of type `Self::Meta` representing the beta parameter of the Beta distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let beta_array = Tensor::<f32>::beta(2.0, 5.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Beta distribution
    /// with alpha = 2.0 and beta = 5.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Beta distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the alpha parameter of the Beta distribution.
    /// - `b`: A value of type `Self::Meta` representing the beta parameter of the Beta distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let beta_array_like = a.beta_like(2.0, 5.0)?;
    /// ```
    /// This example creates an array of random numbers following a Beta distribution,
    /// similar to `a` in shape, with alpha = 2.0 and beta = 5.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Chi-Squared distribution.
    ///
    /// # Parameters
    /// - `df`: A value of type `Self::Meta` representing the degrees of freedom of the Chi-Squared distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let chi_square_array = Tensor::<f32>::chisquare(2.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Chi-Squared distribution
    /// with 2 degrees of freedom.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Chi-Squared distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `df`: A value of type `Self::Meta` representing the degrees of freedom of the Chi-Squared distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let chi_square_array_like = a.chisquare_like(2.0)?;
    /// ```
    /// This example creates an array of random numbers following a Chi-Squared distribution,
    /// similar to `a` in shape, with 2 degrees of freedom.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn chisquare_like(&self, df: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Exponential distribution.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter of the Exponential distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let exponential_array = Tensor::<f32>::exponential(1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following an Exponential distribution
    /// with a rate parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Exponential distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter of the Exponential distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let exponential_array_like = a.exponential_like(1.0)?;
    /// ```
    /// This example creates an array of random numbers following an Exponential distribution,
    /// similar to `a` in shape, with a rate parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Gamma distribution.
    ///
    /// # Parameters
    /// - `shape`: A value of type `Self::Meta` representing the shape parameter of the Gamma distribution.
    /// - `scale`: A value of type `Self::Meta` representing the scale parameter of the Gamma distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let gamma_array = Tensor::<f32>::gamma(2.0, 1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Gamma distribution
    /// with a shape parameter of 2.0 and a scale parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gamma<S: Into<Shape>>(shape: Self::Meta, scale: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Gamma distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `shape`: A value of type `Self::Meta` representing the shape parameter of the Gamma distribution.
    /// - `scale`: A value of type `Self::Meta` representing the scale parameter of the Gamma distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let gamma_array_like = a.gamma_like(2.0, 1.0)?;
    /// ```
    /// This example creates an array of random numbers following a Gamma distribution,
    /// similar to `a` in shape, with a shape parameter of 2.0 and a scale parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Gumbel distribution.
    ///
    /// # Parameters
    /// - `mu`: A value of type `Self::Meta` representing the location parameter of the Gumbel distribution.
    /// - `beta`: A value of type `Self::Meta` representing the scale parameter of the Gumbel distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let gumbel_array = Tensor::<f32>::gumbel(0.0, 1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Gumbel distribution
    /// with a location parameter of 0.0 and a scale parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gumbel<S: Into<Shape>>(mu: Self::Meta, beta: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Gumbel distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mu`: A value of type `Self::Meta` representing the location parameter of the Gumbel distribution.
    /// - `beta`: A value of type `Self::Meta` representing the scale parameter of the Gumbel distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let gumbel_array_like = a.gumbel_like(0.0, 1.0)?;
    /// ```
    /// This example creates an array of random numbers following a Gumbel distribution,
    /// similar to `a` in shape, with a location parameter of 0.0 and a scale parameter of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Log-Normal distribution.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the underlying normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the underlying normal distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let lognormal_array = Tensor::<f32>::lognormal(0.0, 1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Log-Normal distribution
    /// with a mean of 0.0 and a standard deviation of 1.0 for the underlying normal distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn lognormal<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Log-Normal distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the underlying normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the underlying normal distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let lognormal_array_like = a.lognormal_like(0.0, 1.0)?;
    /// ```
    /// This example creates an array of random numbers following a Log-Normal distribution,
    /// similar to `a` in shape, with a mean of 0.0 and a standard deviation of 1.0 for the underlying normal distribution.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Normal (Gaussian) distribution.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the Normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the Normal distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let gaussian_array = Tensor::<f32>::normal_gaussian(0.0, 1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Normal (Gaussian) distribution
    /// with a mean of 0.0 and a standard deviation of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn normal_gaussian<S: Into<Shape>>(mean: Self::Meta, std: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Normal (Gaussian) distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `mean`: A value of type `Self::Meta` representing the mean of the Normal distribution.
    /// - `std`: A value of type `Self::Meta` representing the standard deviation of the Normal distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let gaussian_array_like = a.normal_gaussian_like(0.0, 1.0)?;
    /// ```
    /// This example creates an array of random numbers following a Normal (Gaussian) distribution,
    /// similar to `a` in shape, with a mean of 0.0 and a standard deviation of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Pareto distribution.
    ///
    /// # Parameters
    /// - `pareto_shape`: A value of type `Self::Meta` representing the shape parameter of the Pareto distribution.
    /// - `a`: A value of type `Self::Meta` representing the scale parameter 'a' of the Pareto distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let pareto_array = Tensor::<f32>::pareto(3.0, 1.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Pareto distribution
    /// with a shape parameter of 3.0 and a scale parameter 'a' of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn pareto<S: Into<Shape>>(pareto_shape: Self::Meta, a: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Pareto distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `pareto_shape`: A value of type `Self::Meta` representing the shape parameter of the Pareto distribution.
    /// - `a`: A value of type `Self::Meta` representing the scale parameter 'a' of the Pareto distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let pareto_array_like = a.pareto_like(3.0, 1.0)?;
    /// ```
    /// This example creates an array of random numbers following a Pareto distribution,
    /// similar to `a` in shape, with a shape parameter of 3.0 and a scale parameter 'a' of 1.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Poisson distribution.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter (λ) of the Poisson distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let poisson_array = Tensor::<f32>::poisson(4.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Poisson distribution
    /// with a rate parameter (λ) of 4.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Poisson distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `lambda`: A value of type `Self::Meta` representing the rate parameter (λ) of the Poisson distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let poisson_array_like = a.poisson_like(4.0)?;
    /// ```
    /// This example creates an array of random numbers following a Poisson distribution,
    /// similar to `a` in shape, with a rate parameter (λ) of 4.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Weibull distribution.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the shape parameter of the Weibull distribution.
    /// - `b`: A value of type `Self::Meta` representing the scale parameter of the Weibull distribution.
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let weibull_array = Tensor::<f32>::weibull(1.5, 2.0, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random numbers following a Weibull distribution
    /// with a shape parameter of 1.5 and a scale parameter of 2.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Weibull distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `a`: A value of type `Self::Meta` representing the shape parameter of the Weibull distribution.
    /// - `b`: A value of type `Self::Meta` representing the scale parameter of the Weibull distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let weibull_array_like = a.weibull_like(1.5, 2.0)?;
    /// ```
    /// This example creates an array of random numbers following a Weibull distribution,
    /// similar to `a` in shape, with a shape parameter of 1.5 and a scale parameter of 2.0.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Zipf distribution.
    ///
    /// # Parameters
    /// - `n`: A `u64` value representing the number of elements (size) of the Zipf distribution.
    /// - `a`: A value of type `Self::Meta` representing the exponent parameter of the Zipf distribution.
    /// - `shape`: The shape of the output array, can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let zipf_array = Tensor::<f32>::zipf(1000, 1.5, [1000])?;
    /// ```
    /// This example creates an array of random numbers following a Zipf distribution
    /// with 1000 elements and exponent parameter 1.5.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zipf<S: Into<Shape>>(n: u64, a: Self::Meta, shape: S) -> Result<Self>;

    /// Generates a random number array following the Zipf distribution,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `n`: A `u64` value representing the number of elements (size) of the Zipf distribution.
    /// - `a`: A value of type `Self::Meta` representing the exponent parameter of the Zipf distribution.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([1000])?;
    /// let zipf_array_like = a.zipf_like(1000, 1.5)?;
    /// ```
    /// This example creates an array of random numbers following a Zipf distribution,
    /// similar to `some_instance` in shape, with 1000 elements and exponent parameter 1.5.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn zipf_like(&self, n: u64, a: Self::Meta) -> Result<Self>;

    /// Generates a random number array following the Triangular distribution.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower limit of the distribution.
    /// - `high`: A value of type `Self::Meta` representing the upper limit of the distribution.
    /// - `mode`: A value of type `Self::Meta` representing the mode (peak) of the distribution.
    /// - `shape`: The shape of the output array, can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let triangular_array = Tensor::<f32>::triangular(0, 10, 5, [100])?;
    /// ```
    /// This example creates an array of random numbers following a Triangular distribution
    /// ranging from 0 to 10 with a mode (peak) at 5.
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
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<f32>::randn([100])?;
    /// let triangular_array_like = a.triangular_like(0, 10, 5)?;
    /// ```
    /// This example creates an array of random numbers following a Triangular distribution,
    /// similar to `a` in shape, ranging from 0 to 10 with a mode (peak) at 5.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn triangular_like(&self, low: Self::Meta, high: Self::Meta, mode: Self::Meta) -> Result<Self>;

    #[cfg_attr(feature = "track_caller", track_caller)]
    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self>
        where Self::Meta: IntoScalar<f64>, bool: IntoScalar<Self::Meta>;
}

pub trait RandomInt where Self: Sized {
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// Generates a random integer array with values in the specified range.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower bound of the range (inclusive).
    /// - `high`: A value of type `Self::Meta` representing the upper bound of the range (exclusive).
    /// - `shape`: The shape of the output array, which can be converted from `S` into `Shape`.
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let integer_array = Tensor::<i32>::randint(0, 10, [1000])?;
    /// ```
    /// This example creates a 1000-element array of random integers in the range [0, 10).
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randint<S: Into<Shape>>(low: Self::Meta, high: Self::Meta, shape: S) -> Result<Self>
        where Self::Meta: SampleUniform, <Self::Meta as SampleUniform>::Sampler: Sync;

    /// Generates a random integer array with values in the specified range,
    /// with the same shape as the calling instance.
    ///
    /// # Parameters
    /// - `low`: A value of type `Self::Meta` representing the lower bound of the range (inclusive).
    /// - `high`: A value of type `Self::Meta` representing the upper bound of the range (exclusive).
    ///
    /// # Returns
    /// `Result<Tensor<T>>`: A result containing the generated array or an error.
    ///
    /// # Example
    /// ```
    /// let a = Tensor::<i32>::randn([1000])?;
    /// let integer_array_like = a.randint_like(0, 10)?;
    /// ```
    /// This example creates an array of random integers in the range [0, 10),
    /// similar to `a` in shape.
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self>
        where Self::Meta: SampleUniform, <Self::Meta as SampleUniform>::Sampler: Sync;
}
