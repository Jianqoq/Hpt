use tensor_traits::CommonBounds;
use tensor_types::type_promote::{ Cmp, NormalOut };

use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::reduce::max;

impl<T> _Tensor<T> where T: CommonBounds + NormalOut<T, Output = T> + Cmp {
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<_Tensor<T>> {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = max(self, &[axis], T::ZERO, true, false, None)?;
        let ret = self
            .par_iter()
            .zip(max.par_iter())
            .strided_map(|(a, b)| {
                if a._eq(b) { T::ONE } else { T::ZERO }
            })
            .collect::<_Tensor<T>>();
        Ok(ret)
    }
}

impl<T> Tensor<T> where T: CommonBounds + NormalOut<T, Output = T> + Cmp {
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hardmax(self, axis)?.into()))
    }
}
