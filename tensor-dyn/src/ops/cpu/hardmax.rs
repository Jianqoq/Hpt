use crate::ops::cpu::binary_normal::binary_fn_with_out_simd;
use crate::{tensor::Tensor, tensor_base::_Tensor};
use tensor_traits::CommonBounds;
use tensor_traits::NormalReduce;
use tensor_types::{
    convertion::{Convertor, VecConvertor},
    traits::SimdSelect,
    type_promote::{Cmp, Eval, NormalOut, SimdCmp},
};

impl<T> _Tensor<T>
where
    T: CommonBounds + Cmp<T, Output = bool> + tensor_types::into_scalar::IntoScalar<T> + Convertor,
    bool: NormalOut<T, Output = T>,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: NormalOut<T::Vec, Output = T::Vec>,
    T: Eval<Output = bool>,
    T::Vec: Eval + VecConvertor,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
{
    /// Applies the hardmax function along a specified axis.
    ///
    /// The hardmax function converts the elements along a given axis to binary values, where the largest element
    /// in each slice along the specified axis is set to 1, and all other elements are set to 0. This is similar
    /// to the argmax operation but produces a tensor where the result is in a one-hot encoded form along the
    /// specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the hardmax operation. The elements along this axis are compared,
    ///   and only the largest element in each slice will be set to 1, while the others will be 0.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the hardmax applied along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<_Tensor<T>> {
        let axis = (if axis < 0 {
            (self.layout.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let max = self.max(axis as i64, true)?;
        let ret = {
            use tensor_types::traits::VecTrait;
            binary_fn_with_out_simd(
                self,
                &max,
                |a, max| a._eq(max)._mul(T::ONE),
                |a, max| {
                    let one = T::Vec::splat(T::ONE);
                    a._eq(max)._mul(one)
                },
                None::<_Tensor<T>>,
            )?
        };
        Ok(ret)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + Cmp<T, Output = bool> + tensor_types::into_scalar::IntoScalar<T> + Convertor,
    bool: NormalOut<T, Output = T>,
    T::Vec: SimdCmp,
    <T::Vec as SimdCmp>::Output: NormalOut<T::Vec, Output = T::Vec>,
    T: Eval<Output = bool>,
    T::Vec: Eval + VecConvertor,
    <T::Vec as Eval>::Output: SimdSelect<T::Vec>,
{
    /// Applies the hardmax function along a specified axis.
    ///
    /// The hardmax function converts the elements along a given axis to binary values, where the largest element
    /// in each slice along the specified axis is set to 1, and all other elements are set to 0. This is similar
    /// to the argmax operation but produces a tensor where the result is in a one-hot encoded form along the
    /// specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the hardmax operation. The elements along this axis are compared,
    ///   and only the largest element in each slice will be set to 1, while the others will be 0.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the hardmax applied along the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(
            _Tensor::hardmax(self.inner.as_ref(), axis)?.into(),
        ))
    }
}
