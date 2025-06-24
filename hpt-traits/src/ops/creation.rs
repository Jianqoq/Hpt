use hpt_common::{error::base::TensorError, shape::shape::Shape};
use hpt_types::type_promote::FloatOutBinary;
use hpt_types::{dtype::TypeCommon, into_scalar::Cast, type_promote::NormalOut};
type BoolVector = <bool as TypeCommon>::Vec;

/// A trait defines a set of functions to create tensors.
pub trait TensorCreator
where
    Self: Sized,
{
    /// the output type of the tensor
    type Output;
    /// the meta type of the tensor
    type Meta;
    /// Creates a new uninitialized tensor with the specified shape. The tensor's values will be whatever was in memory at the time of allocation.
    ///
    /// ## Parameters:
    /// `shape`: The desired shape for the tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::empty(&[2, 3])?; // Shape: [2, 3]
    /// ```
    #[track_caller]
    fn empty<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a new tensor of the specified shape, filled with zeros.
    ///
    /// ## Parameters:
    /// `shape`: The desired shape for the tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::zeros(&[2, 3])?; // Shape: [2, 3]
    /// ```
    #[track_caller]
    fn zeros<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a new tensor of the specified shape, filled with ones.
    ///
    /// ## Parameters:
    /// `shape`: The desired shape for the tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::ones(&[2, 3])?; // Shape: [2, 3]
    /// ```
    #[track_caller]
    fn ones<S: Into<Shape>>(shape: S) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Creates a new uninitialized tensor with the same shape as the input tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    /// let b = a.empty_like()?; // Shape: [3, 1]
    /// ```
    #[track_caller]
    fn empty_like(&self) -> Result<Self::Output, TensorError>;

    /// Creates a new zeroed tensor with the same shape as the input tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    /// let b = a.zeros_like()?; // Shape: [3, 1]
    /// ```
    #[track_caller]
    fn zeros_like(&self) -> Result<Self::Output, TensorError>;

    /// Creates a new tensor with all ones with the same shape as the input tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0]).reshape(&[3, 1])?;
    /// let b = a.ones_like()?; // Shape: [3, 1]
    /// ```
    #[track_caller]
    fn ones_like(&self) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Creates a new tensor of the specified shape, filled with a specified value.
    ///
    /// ## Parameters:
    /// `val`: The value to fill the tensor with.
    ///
    /// `shape`: The desired shape for the tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::full(5.0, &[2, 3])?;
    /// // [[5, 5, 5],
    /// //  [5, 5, 5]]
    /// ```
    #[track_caller]
    fn full<S: Into<Shape>>(val: Self::Meta, shape: S) -> Result<Self::Output, TensorError>;

    /// Creates a new tensor filled with a specified value with the same shape as the input tensor.
    ///
    /// ## Parameters:
    /// `val`: The value to fill the tensor with.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2])?;
    /// let b = a.full_like(7.0)?;
    /// // [[7, 7],
    /// //  [7, 7]]
    /// ```
    #[track_caller]
    fn full_like(&self, val: Self::Meta) -> Result<Self::Output, TensorError>;

    /// Creates a 1-D tensor with evenly spaced values within a given interval `[start, end)`.
    ///
    /// ## Parameters:
    /// `start`: Start of interval (inclusive)
    ///
    /// `end`: End of interval (exclusive)
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::arange(0, 5)?; // [0, 1, 2, 3, 4]
    /// let b = Tensor::<f32>::arange(1.5, 5.5)?; // [1.5, 2.5, 3.5, 4.5]
    /// ```
    #[track_caller]
    fn arange<U>(start: U, end: U) -> Result<Self::Output, TensorError>
    where
        usize: Cast<Self::Meta>,
        U: Cast<i64> + Cast<Self::Meta> + Copy;

    /// Creates a 1-D tensor with evenly spaced values within a given interval `[start, end)` with a specified step size.
    ///
    /// ## Parameters:
    /// `start`: Start of interval (inclusive)
    ///
    /// `end`: End of interval (exclusive)
    ///
    /// `step`: Size of spacing between values
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::arange_step(0.0, 5.0, 2.0)?; // [0, 2, 4]
    /// let b = Tensor::<f32>::arange_step(5.0, 0.0, -1.5)?; // [5, 3.5, 2, 0.5]
    /// ```
    #[track_caller]
    fn arange_step(
        start: Self::Meta,
        end: Self::Meta,
        step: Self::Meta,
    ) -> Result<Self::Output, TensorError>
    where
        Self::Meta: Cast<f64> + Cast<f64>,
        f64: Cast<Self::Meta>,
        usize: Cast<Self::Meta>;

    /// Creates a 2-D tensor with ones on the k-th diagonal and zeros elsewhere.
    ///
    /// ## Parameters:
    /// `n`: Number of rows
    ///
    /// `m`: Number of columns
    ///
    /// `k`: Index of the diagonal (0 represents the main diagonal, positive values are above the main diagonal)
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::eye(3, 4, 0)?;
    /// // [[1, 0, 0, 0],
    /// //  [0, 1, 0, 0],
    /// //  [0, 0, 1, 0]]
    /// let b = Tensor::<f32>::eye(3, 4, 1)?;
    /// // [[0, 1, 0, 0],
    /// //  [0, 0, 1, 0],
    /// //  [0, 0, 0, 1]]
    /// ```
    #[track_caller]
    fn eye(n: usize, m: usize, k: usize) -> Result<Self::Output, TensorError>;

    /// Creates a 1-D tensor of `num` evenly spaced values between `start` and `end`.
    ///
    /// ## Parameters:
    /// `start`: The starting value of the sequence
    ///
    /// `end`: The end value of the sequence
    ///
    /// `num`: Number of samples to generate
    ///
    /// `include_end`: Whether to include the end value in the sequence
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::linspace(0.0, 1.0, 5, true)?;
    /// // [0.0, 0.25, 0.5, 0.75, 1.0]
    /// let b = Tensor::<f32>::linspace(0.0, 1.0, 5, false)?;
    /// // [0.0, 0.2, 0.4, 0.6, 0.8]
    /// let c = Tensor::<f32>::linspace(0, 10, 6, true)?;
    /// // [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    /// ```
    #[track_caller]
    fn linspace<U>(
        start: U,
        end: U,
        num: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        U: Cast<f64> + Cast<Self::Meta> + Copy,
        usize: Cast<Self::Meta>,
        f64: Cast<Self::Meta>;

    /// Creates a 1-D tensor with `num` numbers logarithmically spaced between `base^start` and `base^end`.
    ///
    /// ## Parameters:
    /// `start`: The starting value of the sequence (power of base)
    ///
    /// `end`: The end value of the sequence (power of base)
    ///
    /// `num`: Number of samples to generate
    ///
    /// `include_end`: Whether to include the end value in the sequence
    ///
    /// `base`: The base of the log space (default is 10.0)
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::logspace(0.0, 3.0, 4, true, 10.0)?;
    /// // [1.0, 10.0, 100.0, 1000.0]
    /// let b = Tensor::<f32>::logspace(0.0, 3.0, 4, true, 2.0)?;
    /// // [1.0, 2.0, 4.0, 8.0]
    /// let c = Tensor::<f32>::logspace(0.0, 2.0, 4, false, 10.0)?;
    /// // [1.0, 3.1623, 10.0, 31.6228]
    /// ```
    #[track_caller]
    fn logspace<V: Cast<Self::Meta>>(
        start: V,
        end: V,
        num: usize,
        include_end: bool,
        base: V,
    ) -> Result<Self::Output, TensorError>
    where
        Self::Meta: Cast<f64> + num::Float + FloatOutBinary<Self::Meta, Output = Self::Meta>,
        usize: Cast<Self::Meta>,
        f64: Cast<Self::Meta>;

    /// Creates a 1-D tensor with `n` numbers geometrically spaced between `start` and `end`.
    ///
    /// ## Parameters:
    /// `start`: The starting value of the sequence
    ///
    /// `end`: The end value of the sequence
    ///
    /// `num`: Number of samples to generate
    ///
    /// `include_end`: Whether to include the end value in the sequence
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::geomspace(1.0, 1000.0, 4, true)?;
    /// // [1.0, 10.0, 100.0, 1000.0]
    /// let b = Tensor::<f32>::geomspace(1.0, 100.0, 3, false)?;
    /// // [1.0, 4.6416, 21.5443]
    /// let c = Tensor::<f32>::geomspace(1.0, 32.0, 5, true)?;
    /// // [1.0, 2.3784, 5.6569, 13.4543, 32.0000]
    /// ```
    #[track_caller]
    fn geomspace<V: Cast<Self::Meta>>(
        start: V,
        end: V,
        n: usize,
        include_end: bool,
    ) -> Result<Self::Output, TensorError>
    where
        f64: Cast<Self::Meta>,
        usize: Cast<Self::Meta>,
        Self::Meta: Cast<f64> + FloatOutBinary<Self::Meta, Output = Self::Meta>;

    /// Creates a tensor with ones at and below (or above) the k-th diagonal.
    ///
    /// ## Parameters:
    /// `n`: Number of rows
    ///
    /// `m`: Number of columns
    ///
    /// `k`: The diagonal above or below which to fill with ones (0 represents the main diagonal)
    ///
    /// `low_triangle`: If true, fill with ones below and on the k-th diagonal; if false, fill with ones above the k-th diagonal
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::tri(3, 3, 0, true)?;
    /// // [[1, 0, 0],
    /// //  [1, 1, 0],
    /// //  [1, 1, 1]]
    /// let b = Tensor::<f32>::tri(3, 3, 0, false)?;
    /// // [[1, 1, 1],
    /// //  [0, 1, 1],
    /// //  [0, 0, 1]]
    /// let c = Tensor::<f32>::tri(3, 4, 1, true)?;
    /// // [[1, 1, 0, 0],
    /// //  [1, 1, 1, 0],
    /// //  [1, 1, 1, 1]]
    /// ```
    #[track_caller]
    fn tri(n: usize, m: usize, k: i64, low_triangle: bool) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;

    /// Returns a copy of the tensor with elements above the k-th diagonal zeroed.
    ///
    /// ## Parameters:
    /// `k`: Diagonal above which to zero elements. k=0 is the main diagonal, k>0 is above and k<0 is below the main diagonal
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3])?;
    /// let b = a.tril(0)?;
    /// // [[1, 0, 0],
    /// //  [4, 5, 0],
    /// //  [7, 8, 9]]
    /// let c = a.tril(-1)?;
    /// // [[0, 0, 0],
    /// //  [4, 0, 0],
    /// //  [7, 8, 0]]
    /// ```
    #[track_caller]
    fn tril(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<bool, Output = Self::Meta> + Cast<Self::Meta> + TypeCommon,
        <Self::Meta as TypeCommon>::Vec:
            NormalOut<BoolVector, Output = <Self::Meta as TypeCommon>::Vec>;

    /// Returns a copy of the tensor with elements below the k-th diagonal zeroed.
    ///
    /// ## Parameters:
    /// `k`: Diagonal below which to zero elements. k=0 is the main diagonal, k>0 is above and k<0 is below the main diagonal
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3])?;
    /// let b = a.triu(0)?;
    /// // [[1, 2, 3],
    /// //  [0, 5, 6],
    /// //  [0, 0, 9]]
    /// let c = a.triu(1)?;
    /// // [[0, 2, 3],
    /// //  [0, 0, 6],
    /// //  [0, 0, 0]]
    /// ```
    #[track_caller]
    fn triu(&self, k: i64) -> Result<Self::Output, TensorError>
    where
        Self::Meta: NormalOut<bool, Output = Self::Meta> + Cast<Self::Meta> + TypeCommon,
        <Self::Meta as TypeCommon>::Vec:
            NormalOut<BoolVector, Output = <Self::Meta as TypeCommon>::Vec>;

    /// Creates a 2-D identity tensor (1's on the main diagonal and 0's elsewhere).
    ///
    /// ## Parameters:
    /// `n`: Number of rows and columns
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::identity(3)?;
    /// // [[1, 0, 0],
    /// //  [0, 1, 0],
    /// //  [0, 0, 1]]
    /// ```
    #[track_caller]
    fn identity(n: usize) -> Result<Self::Output, TensorError>
    where
        u8: Cast<Self::Meta>;
}
