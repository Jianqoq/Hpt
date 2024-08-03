use crate::{ ops::cpu::reduce::sum, tensor_base::_Tensor };
use tensor_traits::CommonBounds;
use tensor_types::{
    dtype::TypeCommon,
    into_scalar::IntoScalar,
    type_promote::{ Cmp, FloatOut, NormalOut },
};

use super::reduce::max;

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
        let max = max(self, &[axis], T::ZERO, true, false, None)?;
        let exp = self
            .iter()
            .zip(max.iter())
            .strided_map(|(x, y)| { x._sub(y)._exp() })
            .collect::<_Tensor<<T as FloatOut>::Output>>();
        let sum = sum(&exp, &[axis], <T as FloatOut>::Output::ZERO, true, false, None)?;
        Ok(exp / sum)
    }
}
