use crate::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_iterator::{ iterator_traits::ParStridedIteratorZip, TensorIterator };
use tensor_traits::{ CommonBounds, NormalReduce };
use tensor_types::{
    convertion::Convertor,
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ Cmp, FloatOutBinary, FloatOutUnary, NormalOut },
};

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds
{
    /// Applies the softmax function along a specified axis.
    ///
    /// The softmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The softmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j along the specified axis
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the softmax function. The elements along this axis
    ///   will be transformed into probabilities that sum to 1.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the softmax values computed along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn softmax(&self, axis: i64) -> anyhow::Result<_Tensor<<T as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
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
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds
{
    /// Applies the softmax function along a specified axis.
    ///
    /// The softmax function normalizes the elements along the specified axis such that they sum to 1.
    /// It is commonly used in machine learning models, particularly for multi-class classification tasks.
    /// The softmax function transforms each element `x_i` in the input tensor into a probability by computing:
    ///
    /// ```text
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j along the specified axis
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the softmax function. The elements along this axis
    ///   will be transformed into probabilities that sum to 1.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the softmax values computed along
    /// the specified axis.
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn softmax(&self, axis: i64) -> anyhow::Result<Tensor<<T as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: FloatOutBinary<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>
    {
        Ok(Tensor::from(_Tensor::softmax(self, axis)?.into()))
    }
}

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary + Convertor,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    /// Applies the log-softmax function along a specified axis.
    ///
    /// The log-softmax function is the logarithm of the softmax function. It is useful in numerical stability,
    /// particularly in scenarios involving large exponents or probabilities, such as in classification tasks.
    /// The log-softmax function computes the logarithm of the softmax of each element `x_i` as:
    ///
    /// ```text
    /// log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// This function prevents numerical overflow by directly computing the log of the softmax,
    /// rather than computing the softmax first and then taking the log.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the log-softmax function. The elements along this axis
    ///   will be transformed into log-probabilities that correspond to the softmax values.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the log-softmax values computed along
    /// the specified axis.
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
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
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    /// Applies the log-softmax function along a specified axis.
    ///
    /// The log-softmax function is the logarithm of the softmax function. It is useful in numerical stability,
    /// particularly in scenarios involving large exponents or probabilities, such as in classification tasks.
    /// The log-softmax function computes the logarithm of the softmax of each element `x_i` as:
    ///
    /// ```text
    /// log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) for all j along the specified axis
    /// ```
    ///
    /// This function prevents numerical overflow by directly computing the log of the softmax,
    /// rather than computing the softmax first and then taking the log.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply the log-softmax function. The elements along this axis
    ///   will be transformed into log-probabilities that correspond to the softmax values.
    ///
    /// # Returns
    ///
    /// This function returns a `Result` containing a tensor with the log-softmax values computed along
    /// the specified axis.
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            T::Vec: NormalOut<Output = T::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output> +
                Convertor
    {
        Ok(Tensor::from(_Tensor::logsoftmax(self, axis)?.into()))
    }
}
