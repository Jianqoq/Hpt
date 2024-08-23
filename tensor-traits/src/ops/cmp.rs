
use anyhow::Result;
use tensor_types::type_promote::Cmp;

pub trait TensorCmp<T, U> {
    type RHS;
    type Output;

    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    /// 
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.eq(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn neq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;

    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    ///     
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.eq(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn eq<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;
        
    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    /// 
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.lt(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn lt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;

    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    /// 
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.gt(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn gt<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;

    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    /// 
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.le(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn le<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;

    /// check if two tensors are equal, return a bool tensor
    /// 
    /// # Arguments
    /// `rhs` - the tensor to be compared
    /// 
    /// # Returns
    /// bool tensor
    /// 
    /// # Example
    /// ```
    /// use tensor_core::prelude::*;
    /// let a = Tensor::arange(0.0, 100.0).unwrap();
    /// let b = a.ge(0.0).unwrap();
    /// assert_eq!(b.shape(), &[100]);
    /// ```
    #[cfg_attr(feature = "track_caller", track_caller)]
    fn ge<D: Into<Self::RHS>>(&self, rhs: D) -> Result<Self::Output>
        where T: Cmp<U>;
}
