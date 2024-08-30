use crate::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_traits::CommonBounds;
use tensor_types::{
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ Cmp, FloatOutBinary, FloatOutUnary, NormalOut },
};

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds
{
    pub fn softmax(&self, axis: i64) -> anyhow::Result<_Tensor<<T as FloatOutUnary>::Output>>
        where
            <T as TypeCommon>::Vec: NormalOut<Output = <T as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output>
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = self.max(axis as i64, true)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| { x._sub(y)._exp() })
            .collect::<_Tensor<<T as FloatOutUnary>::Output>>();
        let sum = exp.sum(axis as i64, false)?;
        Ok(exp / sum)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds
{
    pub fn softmax(&self, axis: i64) -> anyhow::Result<Tensor<<T as FloatOutUnary>::Output>>
        where
            <T as TypeCommon>::Vec: NormalOut<Output = <T as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output>
    {
        Ok(Tensor::from(_Tensor::softmax(self, axis)?.into()))
    }
}

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            <T as TypeCommon>::Vec: NormalOut<Output = <T as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output>
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = self.max(axis as i64, true)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| { x._sub(y)._exp() })
            .collect::<_Tensor<<T as FloatOutUnary>::Output>>();
        let sum = exp.sum(axis as i64, false)?;
        let ret = exp
            .par_iter()
            .zip(sum.par_iter())
            .strided_map(|(x, y)| { x._div(y)._ln() })
            .collect::<_Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>();
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOutUnary,
        <T as FloatOutUnary>::Output: CommonBounds + TypeCommon,
        <T as FloatOutUnary>::Output: NormalOut +
            NormalOut<<T as FloatOutUnary>::Output, Output = <T as FloatOutUnary>::Output> +
            FloatOutUnary,
        <<T as FloatOutUnary>::Output as FloatOutUnary>::Output: IntoScalar<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output> +
            CommonBounds +
            FloatOutUnary,
        <<<T as FloatOutUnary>::Output as FloatOutUnary>::Output as FloatOutUnary>::Output: CommonBounds
{
    pub fn logsoftmax(
        &self,
        axis: i64
    )
        -> anyhow::Result<Tensor<<<T as FloatOutUnary>::Output as FloatOutUnary>::Output>>
        where
            <T as TypeCommon>::Vec: NormalOut<Output = <T as TypeCommon>::Vec>,
            <<T as FloatOutUnary>::Output as TypeCommon>::Vec: NormalOut<Output = <<T as FloatOutUnary>::Output as TypeCommon>::Vec>,
            <T as FloatOutUnary>::Output: FloatOutBinary<Output = <T as FloatOutUnary>::Output>
    {
        Ok(Tensor::from(_Tensor::logsoftmax(self, axis)?.into()))
    }
}
