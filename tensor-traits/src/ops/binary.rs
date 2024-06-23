use anyhow::Result;

use crate::tensor::{ CommonBounds, TensorInfo, TensorLike };

pub trait NormalBinOps<RHS = Self> {
    type Output;
    type OutputMeta: CommonBounds;
    type InplaceOutput;

    /// Compute addition of `self` and `rhs` element-wise, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f32, 2f32, 3f32]);
    /// let b = Tensor::new([1f32, 2f32, 3f32]);
    /// let c = Tensor::new([2f32, 3f32, 4f32]);
    /// let res = a.add_(b, c).unwrap();
    /// assert_eq!(c, Tensor::new([2f32, 4f32, 6f32]));
    /// ```
    /// # Note
    /// inplace operations is just a suggestion to the backend, the backend can choose to ignore them based on their reference count.
    fn add_<U>(&self, rhs: RHS, out: U) -> Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>;

    /// Compute subtraction of `self` and `rhs` element-wise, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f32, 2f32, 3f32]);
    /// let b = Tensor::new([1f32, 2f32, 3f32]);
    /// let c = Tensor::new([2f32, 3f32, 4f32]);
    /// let res = a.sub_(b, c).unwrap();
    /// assert_eq!(c, Tensor::new([0f32, 0f32, 0f32]));
    /// ```
    /// # Note
    /// inplace operations is just a suggestion to the backend, the backend can choose to ignore them based on their reference count.
    fn sub_<U>(&self, rhs: RHS, out: U) -> Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>;

    /// Compute multiplication of `self` and `rhs` element-wise, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f32, 2f32, 3f32]);
    /// let b = Tensor::new([1f32, 2f32, 3f32]);
    /// let c = Tensor::new([2f32, 3f32, 4f32]);
    /// let res = a.mul_(b, c).unwrap();
    /// assert_eq!(c, Tensor::new([1f32, 4f32, 9f32]));
    /// ```
    /// # Note
    /// inplace operations is just a suggestion to the backend, the backend can choose to ignore them based on their reference count.
    fn mul_<U>(&self, rhs: RHS, out: U) -> Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>;

    /// Compute remainder of `self` and `rhs` element-wise, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([1f32, 2f32, 3f32]);
    /// let b = Tensor::new([1f32, 2f32, 3f32]);
    /// let c = Tensor::new([2f32, 3f32, 4f32]);
    /// let res = a.rem_(b, c).unwrap();
    /// assert_eq!(c, Tensor::new([0f32, 1f32, 0f32]));
    /// ```
    /// # Note
    /// inplace operations is just a suggestion to the backend, the backend can choose to ignore them based on their reference count.
    fn rem_<U>(&self, rhs: RHS, out: U) -> Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>;

    fn convolve(&self, rhs: RHS) -> Result<Self::Output>;
}

pub trait Matmul<RHS = Self> {
    type Output;
    type OutputMeta: CommonBounds;
    type InplaceOutput;
    /// Perform matrix multiplication of `self` and `rhs`, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]);
    /// let b = Tensor::new([[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]);
    /// let c = Tensor::new([[22f32, 28f32], [49f32, 64f32]]);
    /// let res = a.matmul(b).unwrap();
    /// assert_eq!(c, res);
    /// ```
    fn matmul(&self, rhs: RHS) -> Result<Self::Output>;

    /// Inplace Version of matmul. Perform matrix multiplication of `self` and `rhs`, with auto broadcasting.
    ///
    /// # Example
    /// ```rust
    /// use tensor_core::prelude::*;
    /// let a = Tensor::new([[1f32, 2f32, 3f32], [4f32, 5f32, 6f32]]);
    /// let b = Tensor::new([[1f32, 2f32], [3f32, 4f32], [5f32, 6f32]]);
    /// let c = Tensor::new([[22f32, 28f32], [49f32, 64f32]]);
    /// let mut res = Tensor::new([[0f32, 0f32], [0f32, 0f32]]);
    /// a.matmul_(b, &mut res).unwrap();
    /// assert_eq!(c, res);
    /// ```
    fn matmul_<U>(&self, rhs: RHS, out: U) -> Result<Self::Output>
        where
            U: TensorLike<Self::OutputMeta, Output = Self::InplaceOutput> +
                TensorInfo<Self::OutputMeta>;
}
