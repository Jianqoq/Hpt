use std::borrow::BorrowMut;

use hpt_common::error::base::TensorError;
use hpt_types::dtype::TypeCommon;

use crate::tensor::CommonBounds;

/// A trait for binary operations on tensors.
pub trait NormalBinOps<RHS = Self>
where
    <<Self as NormalBinOps<RHS>>::OutputMeta as TypeCommon>::Vec: Send + Sync,
{
    /// The output tensor type.
    type Output;
    /// The output tensor data type.
    type OutputMeta: CommonBounds;
    /// The inplace output tensor type.
    type InplaceOutput;

    /// add with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0]);
    /// let b = Tensor::<f32>::new([3.0]);
    /// let c = a.add_(&b, &mut a.clone())?; // c and a point to the same memory
    /// println!("{}", c); // [5.0]
    /// ```
    fn add_<U>(&self, rhs: RHS, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// subtract with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0]);
    /// let b = Tensor::<f32>::new([3.0]);
    /// let c = a.sub_(&b, &mut a.clone())?; // c and a point to the same memory
    /// println!("{}", c); // [-1.0]
    /// ```
    fn sub_<U>(&self, rhs: RHS, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// multiply with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0]);
    /// let b = Tensor::<f32>::new([3.0]);
    /// let c = a.mul_(&b, &mut a.clone())?; // c and a point to the same memory
    /// println!("{}", c); // [6.0]
    /// ```
    fn mul_<U>(&self, rhs: RHS, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;

    /// rem with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0]);
    /// let b = Tensor::<f32>::new([3.0]);
    /// let c = a.mul_(&b, &mut a.clone())?; // c and a point to the same memory
    /// println!("{}", c); // [6.0]
    /// ```
    fn rem_<U>(&self, rhs: RHS, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>;
}

/// A trait for binary operations on tensors.
pub trait FloatBinOps<RHS = Self> {
    /// The output tensor type.
    type Output;
    /// The output tensor data type.
    type OutputMeta: CommonBounds;
    /// The inplace output tensor type.
    type InplaceOutput;

    /// Compute `sqrt(x^2 + y^2)` for all elements
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let base = Tensor::<f32>::new(&[3.0, 4.0, 5.0]);
    /// let height = Tensor::<f32>::new(&[4.0, 3.0, 12.0]);
    /// let hypotenuse = base.hypot(&height)?; // [5.0, 5.0, 13.0]
    /// let fixed_height = base.hypot(4.0)?; // [5.0, 5.66, 6.40]
    /// ```
    fn hypot<B>(&self, rhs: B) -> std::result::Result<Self::Output, TensorError>
    where
        B: Into<RHS>;

    /// hypot with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let base = Tensor::<f32>::new(&[3.0, 4.0, 5.0]);
    /// let height = Tensor::<f32>::new(&[4.0, 3.0, 12.0]);
    /// let result = base.hypot_(&height, &mut base.clone())?; // [5.0, 5.0, 13.0]
    /// ```
    fn hypot_<B, U>(&self, rhs: B, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        B: Into<RHS>;

    /// division with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new([2.0]);
    /// let b = Tensor::<f32>::new([3.0]);
    /// let c = a.div_(&b, &mut a.clone())?; // [0.6667]
    /// ```
    fn div_<B, U>(&self, rhs: B, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        B: Into<RHS>;

    /// Power of `self` and `rhs`
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[2.0, 3.0, 4.0]);
    /// let b = Tensor::<f32>::new(&[2.0, 3.0, 2.0]);
    /// let c = a.pow(&b)?; // [4.0, 27.0, 16.0]
    /// let d = a.pow(2.0f64)?; // [4.0, 9.0, 16.0]
    /// ```
    fn pow<B>(&self, rhs: B) -> std::result::Result<Self::Output, TensorError>
    where
        B: Into<RHS>;

    /// Power of `self` and `rhs` with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// let a = Tensor::<f32>::new(&[2.0, 3.0, 4.0]);
    /// let b = Tensor::<f32>::new(&[2.0, 3.0, 2.0]);
    /// let c = a.pow_(&b, &mut a.clone())?; // [4.0, 27.0, 16.0]
    /// ```
    fn pow_<B, U>(&self, rhs: B, out: U) -> std::result::Result<Self::Output, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput>,
        B: Into<RHS>;
}

/// A trait for matrix multiplication operations on tensors.
pub trait Matmul<RHS = Self>
where
    <<Self as Matmul<RHS>>::OutputMeta as TypeCommon>::Vec: Send + Sync,
{
    /// The output tensor type.
    type Output;
    /// The output tensor data type.
    type OutputMeta: CommonBounds;
    /// The inplace output tensor type.
    type InplaceOutput;

    /// Perform matrix multiplication of two tensors. The behavior depends on the dimensions of the input tensors:
    ///
    /// - If both tensors are 2D, they are multiplied as matrices
    /// - If either tensor is ND (N > 2), it is treated as a stack of matrices
    /// - Broadcasting is applied to match dimensions
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.matmul(&b)?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.matmul(&e)?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn matmul(&self, rhs: RHS) -> std::result::Result<Self::Output, TensorError>;

    /// matrix multiplication with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.matmul_(&b, &mut a.clone())?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.matmul_(&e, &mut d.clone())?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn matmul_<U>(&self, rhs: RHS, out: U) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>;
}

/// A trait for matrix multiplication operations on tensors with post-operation.
pub trait MatmulPost<RHS = Self>
where
    <<Self as MatmulPost<RHS>>::OutputMeta as TypeCommon>::Vec: Send + Sync,
{
    /// The output tensor type.
    type Output;
    /// The output tensor data type.
    type OutputMeta: CommonBounds;
    /// The inplace output tensor type.
    type InplaceOutput;

    /// Same as `matmul` but will perform post operation before writing final result to the memory.
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `post_op`: post operation function that takes scalar `T` as input and returns scalar `T`
    ///
    /// `post_op_vec`: post operation function that takes simd vector `T::Simd` as input and returns simd vector `T::Simd`
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.matmul_post(&b, |x| x._relu(), |x| x._relu())?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.matmul_post(&e, |x| x._relu(), |x| x._relu())?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn matmul_post(
        &self,
        rhs: RHS,
        post_op: fn(Self::OutputMeta) -> Self::OutputMeta,
        post_op_vec: fn(
            <<Self as MatmulPost<RHS>>::OutputMeta as TypeCommon>::Vec,
        ) -> <<Self as MatmulPost<RHS>>::OutputMeta as TypeCommon>::Vec,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// matrix multiplication with specified output tensor and post operation
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `out`: The output tensor.
    ///
    /// `post_op`: post operation function that takes scalar `T` as input and returns scalar `T`
    ///
    /// `post_op_vec`: post operation function that takes simd vector `T::Simd` as input and returns simd vector `T::Simd`
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.matmul_post_(&b, &mut a.clone(), |x| x._relu(), |x| x._relu())?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.matmul_post_(&e, &mut d.clone(), |x| x._relu(), |x| x._relu())?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn matmul_post_<U>(
        &self,
        rhs: RHS,
        post_op: fn(Self::OutputMeta) -> Self::OutputMeta,
        post_op_vec: fn(
            <<Self as MatmulPost<RHS>>::OutputMeta as TypeCommon>::Vec,
        ) -> <<Self as MatmulPost<RHS>>::OutputMeta as TypeCommon>::Vec,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>;
}
/// A trait for gemm operations on tensors.
pub trait Gemm<RHS = Self>
where
    <<Self as Gemm<RHS>>::OutputMeta as TypeCommon>::Vec: Send + Sync,
{
    /// The output tensor type.
    type Output;
    /// The output tensor data type.
    type OutputMeta: CommonBounds;
    /// The inplace output tensor type.
    type InplaceOutput;

    /// Perform gemm (general matrix multiplication) of two tensors. The behavior depends on the dimensions of the input tensors:
    ///
    /// - If both tensors are 2D, they are multiplied as matrices
    /// - If either tensor is ND (N > 2), it is treated as a stack of matrices
    /// - Broadcasting is applied to match dimensions
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `alpha`: Scaling factor for the matrix product (A @ B)
    ///
    /// `beta`: Scaling factor for the existing values in output matrix C
    ///
    /// `conj_dst`: Whether to conjugate C before scaling with beta
    ///
    /// `conj_lhs`: Whether to conjugate A before multiplication
    ///
    /// `conj_rhs`: Whether to conjugate B before multiplication
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.gemm(&b, 0.0, 1.0, false, false, false)?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.gemm(&e, 0.0, 1.0, false, false, false)?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn gemm(
        &self,
        rhs: RHS,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
    ) -> std::result::Result<Self::Output, TensorError>;

    /// gemm (general matrix multiplication) with specified output tensor
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `alpha`: Scaling factor for the matrix product (A @ B)
    ///
    /// `beta`: Scaling factor for the existing values in output matrix C
    ///
    /// `conj_dst`: Whether to conjugate C before scaling with beta
    ///
    /// `conj_lhs`: Whether to conjugate A before multiplication
    ///
    /// `conj_rhs`: Whether to conjugate B before multiplication
    ///
    /// `out`: The output tensor.
    ///
    /// ## Example:
    /// ```rust
    /// // 2D matrix multiplication
    /// let a = Tensor::<f64>::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::<f64>::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.gemm_(&b, 0.0, 1.0, false, false, false, &mut a.clone())?;
    /// println!("2D result:\n{}", c);
    ///
    /// // 3D batch matrix multiplication
    /// let d = Tensor::<f64>::ones(&[2, 2, 3])?; // 2 matrices of shape 2x3
    /// let e = Tensor::<f64>::ones(&[2, 3, 2])?; // 2 matrices of shape 3x2
    /// let f = d.gemm_(&e, 0.0, 1.0, false, false, false, &mut d.clone())?; // 2 matrices of shape 2x2
    /// println!("3D result:\n{}", f);
    /// ```
    #[track_caller]
    fn gemm_<U>(
        &self,
        rhs: RHS,
        alpha: Self::OutputMeta,
        beta: Self::OutputMeta,
        conj_dst: bool,
        conj_lhs: bool,
        conj_rhs: bool,
        out: U,
    ) -> std::result::Result<Self::InplaceOutput, TensorError>
    where
        U: BorrowMut<Self::InplaceOutput> + BorrowMut<Self::InplaceOutput>;
}

/// A trait for tensor dot operations on tensors.
pub trait TensorDot<RHS = Self> {
    /// The output tensor type.
    type Output;

    /// Compute tensor dot product along specified axes. This is a generalization of matrix multiplication to higher dimensions.
    ///
    /// ## Parameters:
    /// `rhs`: The right-hand side tensor.
    ///
    /// `axes`: A tuple of two arrays specifying the axes to contract over:
    /// - First array contains axes from the first tensor
    /// - Second array contains axes from the second tensor
    /// - Arrays must have same length N
    ///
    /// ## Example:
    /// ```rust
    /// // Matrix multiplication (2D tensordot)
    /// let a = Tensor::new(&[[1., 2.], [3., 4.]]);
    /// let b = Tensor::new(&[[5., 6.], [7., 8.]]);
    /// let c = a.tensordot(&b, ([1], [0]))?; // Contract last axis of a with first axis of b
    /// println!("Matrix multiplication:\n{}", c);
    ///
    /// // Higher dimensional example
    /// let d = Tensor::<f32>::ones(&[2, 3, 4])?;
    /// let e = Tensor::<f32>::ones(&[4, 3, 2])?;
    /// let f = d.tensordot(&e, ([1, 2], [1, 0]))?; // Contract axes 1,2 of d with axes 1,0 of e
    /// println!("Higher dimensional result:\n{}", f);
    /// ```
    fn tensordot<const N: usize>(
        &self,
        rhs: &RHS,
        axes: ([i64; N], [i64; N]),
    ) -> std::result::Result<Self::Output, TensorError>;
}
