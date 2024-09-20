use crate::{tensor::Tensor, tensor_base::_Tensor};
use tensor_iterator::TensorIterator;
use tensor_traits::CommonBounds;
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{Cmp, FloatOutBinary, FloatOutUnary, NormalOut},
};

impl<T> _Tensor<T>
where
    T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
    <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
    <T as FloatOutUnary>::Output: NormalOut
        + NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output>
        + FloatOutUnary,
    <<T as FloatOutUnary>::Output as FloatOutUnary>::Output:
        IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> + CommonBounds,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn softmax(&self, axis: i64) -> anyhow::Result<_Tensor<<T as FloatOutUnary>::Output>>
    where
        T::Vec: NormalOut<Output = T::Vec>,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
        <T as FloatOutUnary>::Output:
            FloatOutBinary<Output = <T as FloatOutUnary>::Output> + Convertor,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
    {
        let axis = (if axis < 0 {
            (self.layout.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let max = self.max(axis as i64, true)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| x._sub(y)._exp())
            .collect::<_Tensor<<T as FloatOutUnary>::Output>>();
        let sum = exp.sum(axis as i64, false)?;
        Ok(exp / sum)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
    <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
    <T as FloatOutUnary>::Output: NormalOut
        + NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output>
        + FloatOutUnary,
    <<T as FloatOutUnary>::Output as FloatOutUnary>::Output:
        IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> + CommonBounds,
{
    /// Applies the Softmax function to the tensor along the specified axis.
    ///
    /// The `softmax` function computes the softmax of each element along the specified axis of the tensor.
    /// The softmax function is often used in multi-class classification problems to convert logits into probabilities.
    ///
    /// The softmax of an element `x_i` is given by:
    /// `softmax(x_i) = exp(x_i) / sum(exp(x_j))` for all `j` along the specified axis.
    ///
    /// # Parameters
    ///
    /// - `axis`: The axis along which to compute the softmax function.
    ///
    /// # Returns
    ///
    /// - A new tensor with the softmax values along the specified axis.
    ///
    /// # Notes
    ///
    /// - **Normalization**: The elements are exponentiated and then normalized by dividing by the sum of exponentials.
    /// - **Axis Specification**: The softmax is computed along the given axis, treating the other axes independently.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn softmax(&self, axis: i64) -> anyhow::Result<Tensor<<T as FloatOutUnary>::Output>>
    where
        T::Vec: NormalOut<Output = T::Vec>,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
        <T as FloatOutUnary>::Output:
            FloatOutBinary<Output = <T as FloatOutUnary>::Output> + Convertor,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
    {
        Ok(Tensor::from(_Tensor::softmax(self, axis)?.into()))
    }
}

impl<T> _Tensor<T>
where
    T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
    <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
    <T as FloatOutUnary>::Output: NormalOut
        + NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output>
        + FloatOutUnary,
    <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>
        + CommonBounds
        + FloatOutUnary,
    <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output:
        CommonBounds,
{
    pub fn logsoftmax(
        &self,
        axis: i64,
    ) -> anyhow::Result<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
    where
        T::Vec: NormalOut<Output = T::Vec>,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
        <T as FloatOutUnary>::Output:
            FloatOutBinary<Output = <T as FloatOutUnary>::Output> + Convertor,
    {
        let axis = (if axis < 0 {
            (self.layout.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let max = self.max(axis as i64, true)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| x._sub(y)._exp())
            .collect::<_Tensor<<T as FloatOutUnary>::Output>>();
        let sum = exp.sum(axis as i64, false)?;
        let ret = exp
            .par_iter()
            .zip(sum.par_iter())
            .strided_map(|(x, y)| x._div(y)._ln())
            .collect::<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>();
        Ok(ret)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
    <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
    <T as FloatOutUnary>::Output: NormalOut
        + NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output>
        + FloatOutUnary,
    <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>
        + CommonBounds
        + FloatOutUnary,
    <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output:
        CommonBounds,
{
    pub fn logsoftmax(
        &self,
        axis: i64,
    ) -> anyhow::Result<Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
    where
        T::Vec: NormalOut<Output = T::Vec>,
        <<T as FloatOutUnary>::Output as TypeCommon>::Vec:
            NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
        <T as FloatOutUnary>::Output:
            FloatOutBinary<Output = <T as FloatOutUnary>::Output> + Convertor,
    {
        Ok(Tensor::from(_Tensor::logsoftmax(self, axis)?.into()))
    }
}
