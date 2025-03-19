use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_types::into_scalar::Cast;
use rand_distr::uniform::SampleUniform;

/// A trait for generating random numbers.
pub trait Random
where
    Self: Sized,
{
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// create a Tensor with data in normal distribution. `mean = 0.0`, `std_dev = 1.0`.
    ///
    /// ## Parameters:
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::randn([10, 10])?;
    /// ```
    #[track_caller]
    fn randn<S: Into<Shape>>(shape: S) -> Result<Self, TensorError>;

    /// same as `randn` but the shape will be based on `x`.
    ///
    /// ## Parameters:
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::randn([10, 10])?;
    /// let b = a.randn_like()?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn randn_like(&self) -> Result<Self, TensorError>;

    /// create a Tensor with data uniformly distributed between `low` and `high`.
    ///
    /// ## Parameters:
    /// `shape`: shape of the output
    ///
    /// `low`: the lowest value
    ///
    /// `high`: the highest value
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::rand([10, 10], 0.0, 10.0)?;
    /// ```
    #[track_caller]
    fn rand<S: Into<Shape>>(
        shape: S,
        low: Self::Meta,
        high: Self::Meta,
    ) -> Result<Self, TensorError>;

    /// same as `rand` but the shape will be based on `x`.
    ///
    /// ## Parameters:
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::rand([10, 10], 0.0, 10.0)?;
    /// let b = a.rand_like(0.0, 10.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn rand_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a beta distribution with parameters `alpha` and `beta`.
    /// The beta distribution is a continuous probability distribution defined on the interval [0, 1].
    ///
    /// ## Parameters:
    /// `alpha`: Shape parameter alpha (α) of the beta distribution. Must be positive.
    ///
    /// `beta`: Shape parameter beta (β) of the beta distribution. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::beta(2.0, 5.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn beta<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S) -> Result<Self, TensorError>;

    /// Same as `beta` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a beta distribution with parameters `alpha` and `beta`.
    ///
    /// ## Parameters:
    /// `alpha`: Shape parameter alpha (α) of the beta distribution. Must be positive.
    ///
    /// `beta`: Shape parameter beta (β) of the beta distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let b = x.beta_like(2.0, 5.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn beta_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a chi-square distribution with `df` degrees of freedom.
    /// The chi-square distribution is a continuous probability distribution of the sum of squares of `df` independent standard normal random variables.
    ///
    /// ## Parameters:
    /// `df`: Degrees of freedom parameter. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::chisquare(5.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn chisquare<S: Into<Shape>>(df: Self::Meta, shape: S) -> Result<Self, TensorError>;

    /// Same as `chisquare` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a chi-square distribution with `df` degrees of freedom.
    ///
    /// ## Parameters:
    /// `df`: Degrees of freedom parameter. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let b = x.chisquare_like(5.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn chisquare_like(&self, df: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from an exponential distribution with rate parameter `lambda`.
    /// The exponential distribution describes the time between events in a Poisson point process.
    ///
    /// ## Parameters:
    /// `lambda`: Rate parameter (λ) of the exponential distribution. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::exponential(2.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn exponential<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError>;

    /// Same as `exponential` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from an exponential distribution with rate parameter `lambda`.
    ///
    /// ## Parameters:
    /// `lambda`: Rate parameter (λ) of the exponential distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let b = x.exponential_like(2.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn exponential_like(&self, lambda: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a gamma distribution with shape parameter `shape_param` (often denoted as k or α) and scale parameter `scale` (often denoted as θ).
    /// The gamma distribution is a continuous probability distribution that generalizes the exponential distribution.
    ///
    /// ## Parameters:
    /// `shape_param`: Shape parameter (k or α) of the gamma distribution. Must be positive.
    ///
    /// `scale`: Scale parameter (θ) of the gamma distribution. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::gamma(2.0, 2.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn gamma<S: Into<Shape>>(
        shape: Self::Meta,
        scale: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `gamma` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a gamma distribution with shape parameter `shape_param` and scale parameter `scale`.
    ///
    /// ## Parameters:
    /// `shape_param`: Shape parameter (k or α) of the gamma distribution. Must be positive.
    ///
    /// `scale`: Scale parameter (θ) of the gamma distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let g = x.gamma_like(2.0, 2.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn gamma_like(&self, shape: Self::Meta, scale: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Gumbel distribution (also known as the Extreme Value Type I distribution) with location parameter `mu` and scale parameter `beta`.
    /// The Gumbel distribution is commonly used to model the distribution of extreme values.
    ///
    /// ## Parameters:
    /// `mu`: Location parameter (μ) of the Gumbel distribution.
    ///
    /// `beta`: Scale parameter (β) of the Gumbel distribution. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::gumbel(0.0, 1.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn gumbel<S: Into<Shape>>(
        mu: Self::Meta,
        beta: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `gumbel` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a Gumbel distribution with location parameter `mu` and scale parameter `beta`.
    ///
    /// ## Parameters:
    /// `mu`: Location parameter (μ) of the Gumbel distribution.
    ///
    /// `beta`: Scale parameter (β) of the Gumbel distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let g = x.gumbel_like(0.0, 1.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn gumbel_like(&self, mu: Self::Meta, beta: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a log-normal distribution. A random variable is log-normally distributed if the logarithm of the random variable is normally distributed.
    /// The parameters `mean` and `std` are the mean and standard deviation of the underlying normal distribution.
    ///
    /// ## Parameters:
    /// `mean`: Mean (μ) of the underlying normal distribution.
    ///
    /// `std`: Standard deviation (σ) of the underlying normal distribution. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::lognormal(0.0, 1.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn lognormal<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `lognormal` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a log-normal distribution with parameters `mean` and `std` of the underlying normal distribution.
    ///
    /// ## Parameters:
    /// `mean`: Mean (μ) of the underlying normal distribution.
    ///
    /// `std`: Standard deviation (σ) of the underlying normal distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let l = x.lognormal_like(0.0, 1.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn lognormal_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a normal (Gaussian) distribution with specified mean and standard deviation.
    /// The normal distribution is a continuous probability distribution that is symmetric around its mean, showing the familiar bell-shaped curve.
    ///
    /// ## Parameters:
    /// `mean`: Mean (μ) of the distribution, determining the center of the bell curve.
    ///
    /// `std`: Standard deviation (σ) of the distribution, determining the spread. Must be positive.
    ///
    /// `shape`: shape of the output
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::normal_gaussian(0.0, 1.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn normal_gaussian<S: Into<Shape>>(
        mean: Self::Meta,
        std: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `normal_gaussian` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a normal distribution with specified mean and standard deviation.
    ///
    /// ## Parameters:
    /// `mean`: Mean (μ) of the distribution, determining the center of the bell curve.
    ///
    /// `std`: Standard deviation (σ) of the distribution, determining the spread. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let n = x.normal_gaussian_like(0.0, 1.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn normal_gaussian_like(&self, mean: Self::Meta, std: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Pareto distribution.
    /// The Pareto distribution is a power-law probability distribution often used to describe the distribution of wealth, population sizes, and many other natural and social phenomena.
    ///
    /// ## Parameters:
    /// `scale`: Scale parameter (xₘ), also known as the minimum possible value. Must be positive.
    ///
    /// `shape`: Shape parameter (α), also known as the Pareto index. Must be positive.
    ///
    /// `tensor_shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::pareto(1.0, 3.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn pareto<S: Into<Shape>>(
        pareto_shape: Self::Meta,
        a: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `pareto` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a Pareto distribution with specified scale and shape parameters.
    ///
    /// ## Parameters:
    /// `scale`: Scale parameter (xₘ), also known as the minimum possible value. Must be positive.
    ///
    /// `shape`: Shape parameter (α), also known as the Pareto index. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let p = x.pareto_like(1.0, 3.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn pareto_like(&self, pareto_shape: Self::Meta, a: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Poisson distribution.
    /// The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, assuming these events occur with a known constant mean rate (λ) and independently of the time since the last event.
    ///
    /// ## Parameters:
    /// `lambda`: Rate parameter (λ) of the Poisson distribution. Must be positive.
    ///
    /// `shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::poisson(5.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn poisson<S: Into<Shape>>(lambda: Self::Meta, shape: S) -> Result<Self, TensorError>;

    /// Same as `poisson` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a Poisson distribution with specified rate parameter.
    ///
    /// ## Parameters:
    /// `lambda`: Rate parameter (λ) of the Poisson distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let p = x.poisson_like(5.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn poisson_like(&self, lambda: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Weibull distribution.
    /// The Weibull distribution is a continuous probability distribution commonly used in reliability engineering, survival analysis, and extreme value theory.
    ///
    /// ## Parameters:
    /// `shape`: Shape parameter (k), determines the shape of the distribution. Must be positive.
    ///
    /// `scale`: Scale parameter (λ), determines the spread of the distribution. Must be positive.
    ///
    /// `tensor_shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::weibull(2.0, 1.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn weibull<S: Into<Shape>>(a: Self::Meta, b: Self::Meta, shape: S)
        -> Result<Self, TensorError>;

    /// Same as `weibull` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a Weibull distribution with specified shape and scale parameters.
    ///
    /// ## Parameters:
    /// `shape`: Shape parameter (k), determines the shape of the distribution. Must be positive.
    ///
    /// `scale`: Scale parameter (λ), determines the spread of the distribution. Must be positive.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let w = x.weibull_like(2.0, 1.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn weibull_like(&self, a: Self::Meta, b: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Zipf distribution.
    /// The Zipf distribution is a discrete probability distribution commonly used to model frequency distributions of ranked data in various physical and social phenomena, where the frequency of any element is inversely proportional to its rank.
    ///
    /// ## Parameters:
    /// `n`: Number of elements (N). Defines the range of possible values [1, N].
    ///
    /// `s`: Exponent parameter (s). Controls the skewness of the distribution. Must be greater than 1.
    ///
    /// `shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zipf(1000.0, 2.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn zipf<S: Into<Shape>>(n: Self::Meta, a: Self::Meta, shape: S) -> Result<Self, TensorError>;

    /// Same as `zipf` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a Zipf distribution with specified number of elements and exponent parameter.
    ///
    /// ## Parameters:
    /// `n`: Number of elements (N). Defines the range of possible values [1, N].
    ///
    /// `s`: Exponent parameter (s). Controls the skewness of the distribution. Must be greater than 1.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let z = x.zipf_like(1000.0, 2.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn zipf_like(&self, n: Self::Meta, a: Self::Meta) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a triangular distribution.
    /// The triangular distribution is a continuous probability distribution with a lower limit `low`, upper limit `high`, and mode `mode`. It forms a triangular shape in its probability density function.
    ///
    /// ## Parameters:
    /// `low`: Lower limit (a) of the distribution.
    ///
    /// `high`: Upper limit (b) of the distribution. Must be greater than `low`.
    ///
    /// `mode`: Mode (c) of the distribution. Must be between `low` and `high`.
    ///
    /// `shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::triangular(0.0, 10.0, 5.0, &[10, 10])?;
    /// ```
    #[track_caller]
    fn triangular<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>;

    /// Same as `triangular` but the shape will be based on `x`.
    /// Creates a Tensor with values drawn from a triangular distribution with specified lower limit, upper limit, and mode.
    ///
    /// ## Parameters:
    /// `low`: Lower limit (a) of the distribution.
    ///
    /// `high`: Upper limit (b) of the distribution. Must be greater than `low`.
    ///
    /// `mode`: Mode (c) of the distribution. Must be between `low` and `high`.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<f32>::randn(&[10, 10])?;
    /// let t = x.triangular_like(0.0, 10.0, 5.0)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn triangular_like(
        &self,
        low: Self::Meta,
        high: Self::Meta,
        mode: Self::Meta,
    ) -> Result<Self, TensorError>;

    /// Create a Tensor with values drawn from a Bernoulli distribution.
    /// The Bernoulli distribution is a discrete probability distribution of a random variable which takes the value 1 with probability p and the value 0 with probability q = 1 - p.
    ///
    /// ## Parameters:
    /// `shape`: Shape of the output tensor.
    ///
    /// `p`: Success probability (p). Must be in the interval [0, 1].
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::bernoulli(&[10, 10], 0.7)?;
    /// ```
    #[track_caller]
    fn bernoulli<S: Into<Shape>>(shape: S, p: Self::Meta) -> Result<Self, TensorError>
    where
        Self::Meta: Cast<f64>,
        bool: Cast<Self::Meta>;
}

/// A trait for generating random integers.
pub trait RandomInt
where
    Self: Sized,
{
    /// Associated type for meta-information or parameters relevant to distributions.
    type Meta;

    /// Create a Tensor with random integers drawn uniformly from the half-open interval `[low, high)`.
    /// The distribution is uniform, meaning each integer in the range has an equal probability of being drawn.
    ///
    /// ## Parameters:
    /// `low`: Lower bound (inclusive) of the range.
    ///
    /// `high`: Upper bound (exclusive) of the range.
    ///
    /// `shape`: Shape of the output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<i32>::randint(0, 100, &[10, 10])?;
    /// ```
    #[track_caller]
    fn randint<S: Into<Shape>>(
        low: Self::Meta,
        high: Self::Meta,
        shape: S,
    ) -> Result<Self, TensorError>
    where
        Self::Meta: SampleUniform,
        <Self::Meta as SampleUniform>::Sampler: Sync;

    /// Same as `randint` but the shape will be based on `x`.
    /// Creates a Tensor with random integers drawn uniformly from the half-open interval `[low, high)`.
    ///
    /// ## Parameters:
    /// `low`: Lower bound (inclusive) of the range.
    ///
    /// `high`: Upper bound (exclusive) of the range.
    ///
    /// ## Example:
    /// ```rust
    /// let x = Tensor::<i32>::randn(&[10, 10])?;
    /// let r = x.randint_like(0, 100)?; // shape: [10, 10]
    /// ```
    #[track_caller]
    fn randint_like(&self, low: Self::Meta, high: Self::Meta) -> Result<Self, TensorError>
    where
        Self::Meta: SampleUniform,
        <Self::Meta as SampleUniform>::Sampler: Sync;
}
