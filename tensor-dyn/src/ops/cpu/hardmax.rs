use tensor_traits::CommonBounds;
use tensor_types::{ dtype::TypeCommon, into_vec::IntoVec, type_promote::{ Cmp, NormalOut } };

use crate::{ tensor::Tensor, tensor_base::_Tensor };

use super::reduce::reduce;

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + tensor_types::into_scalar::IntoScalar<T>
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<_Tensor<T>>
        where <T as TypeCommon>::Vec: IntoVec<<T as TypeCommon>::Vec>
    {
        // let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        // let max = reduce(
        //     self,
        //     |a, b| a._max(b),
        //     |a, b| todo!(),
        //     &[axis],
        //     T::ZERO,
        //     true,
        //     false,
        //     None
        // )?;
        // let ret = self
        //     .par_iter()
        //     .zip(max.par_iter())
        //     .strided_map(
        //         |(a, b)| {
        //             if a._eq(b) { T::ONE } else { T::ZERO }
        //         },
        //         |(a, b)| { todo!() }
        //     )
        //     .collect::<_Tensor<T>>();
        // Ok(ret)
        todo!()
    }
}

impl<T> Tensor<T> where T: CommonBounds + NormalOut<T, Output = T> + Cmp + tensor_types::into_scalar::IntoScalar<T> {
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hardmax(self, axis)?.into()))
    }
}
