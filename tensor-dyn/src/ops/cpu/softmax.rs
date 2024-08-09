use crate::{ tensor::Tensor, tensor_base::_Tensor };
use tensor_traits::CommonBounds;
use tensor_types::{
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ Cmp, FloatOut, NormalOut },
};

use super::reduce::reduce;

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOut,
        <T as FloatOut>::Output: CommonBounds + TypeCommon,
        <T as FloatOut>::Output: NormalOut +
            NormalOut<<T as FloatOut>::Output, Output = <T as FloatOut>::Output> +
            FloatOut,
        <<T as FloatOut>::Output as FloatOut>::Output: IntoScalar<<<T as FloatOut>::Output as FloatOut>::Output> +
            CommonBounds
{
    pub fn softmax(
        &self,
        axis: i64
    ) -> anyhow::Result<_Tensor<<<T as FloatOut>::Output as FloatOut>::Output>> {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = reduce(self, |a, b| a._max(b), &[axis], T::NEG_INF, true, false, None)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| { x._sub(y)._exp() })
            .collect::<_Tensor<<T as FloatOut>::Output>>();
        let sum = reduce(
            &exp,
            |a, b| a._add(b),
            &[axis],
            <T as FloatOut>::Output::ZERO,
            true,
            false,
            None
        )?;
        Ok(exp / sum)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOut,
        <T as FloatOut>::Output: CommonBounds + TypeCommon,
        <T as FloatOut>::Output: NormalOut +
            NormalOut<<T as FloatOut>::Output, Output = <T as FloatOut>::Output> +
            FloatOut,
        <<T as FloatOut>::Output as FloatOut>::Output: IntoScalar<<<T as FloatOut>::Output as FloatOut>::Output> +
            CommonBounds
{
    pub fn softmax(
        &self,
        axis: i64
    ) -> anyhow::Result<Tensor<<<T as FloatOut>::Output as FloatOut>::Output>> {
        Ok(Tensor::from(_Tensor::softmax(self, axis)?.into()))
    }
}

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOut,
        <T as FloatOut>::Output: CommonBounds + TypeCommon,
        <T as FloatOut>::Output: NormalOut +
            NormalOut<<T as FloatOut>::Output, Output = <T as FloatOut>::Output> +
            FloatOut,
        <<T as FloatOut>::Output as FloatOut>::Output: IntoScalar<<<T as FloatOut>::Output as FloatOut>::Output> +
            CommonBounds +
            FloatOut,
        <<<T as FloatOut>::Output as FloatOut>::Output as FloatOut>::Output: CommonBounds
{
    pub fn logsoftmax(
        &self,
        axis: i64
    ) -> anyhow::Result<_Tensor<<<<T as FloatOut>::Output as FloatOut>::Output as FloatOut>::Output>> {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = reduce(self, |a, b| a._max(b), &[axis], T::NEG_INF, true, false, None)?;
        let exp = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(x, y)| { x._sub(y)._exp() })
            .collect::<_Tensor<<T as FloatOut>::Output>>();
        let sum = reduce(
            &exp,
            |a, b| a._add(b),
            &[axis],
            <T as FloatOut>::Output::ZERO,
            true,
            false,
            None
        )?;
        let ret = exp
            .par_iter()
            .zip(sum.par_iter())
            .strided_map(|(x, y)| { x._div(y)._ln() })
            .collect::<_Tensor<<<<T as FloatOut>::Output as FloatOut>::Output as FloatOut>::Output>>();
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + FloatOut,
        <T as FloatOut>::Output: CommonBounds + TypeCommon,
        <T as FloatOut>::Output: NormalOut +
            NormalOut<<T as FloatOut>::Output, Output = <T as FloatOut>::Output> +
            FloatOut,
        <<T as FloatOut>::Output as FloatOut>::Output: IntoScalar<<<T as FloatOut>::Output as FloatOut>::Output> +
            CommonBounds +
            FloatOut,
        <<<T as FloatOut>::Output as FloatOut>::Output as FloatOut>::Output: CommonBounds
{
    pub fn logsoftmax(
        &self,
        axis: i64
    ) -> anyhow::Result<Tensor<<<<T as FloatOut>::Output as FloatOut>::Output as FloatOut>::Output>> {
        Ok(Tensor::from(_Tensor::logsoftmax(self, axis)?.into()))
    }
}
