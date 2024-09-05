use tensor_traits::CommonBounds;
use tensor_types::{
    dtype::TypeCommon,
    into_vec::IntoVec,
    type_promote::{ Cmp, NormalOut, SimdCmp },
};
use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + tensor_types::into_scalar::IntoScalar<T>,
        bool: NormalOut<T, Output = T>,
        <T as TypeCommon>::Vec: SimdCmp + NormalOut<Output = <T as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp>::Output: NormalOut<
            <T as TypeCommon>::Vec,
            Output = <T as TypeCommon>::Vec
        >
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<_Tensor<T>>
        where <T as TypeCommon>::Vec: IntoVec<<T as TypeCommon>::Vec>
    {
        let axis = (if axis < 0 { (self.layout.ndim() as i64) + axis } else { axis }) as usize;
        let max = self.max(axis as i64, true)?;
        #[cfg(feature = "simd")]
        let ret = {
            self.par_iter_simd()
                .zip(max.par_iter_simd())
                .strided_map(
                    |(res, (a, b))| {
                        *res = a._eq(b)._mul(T::ONE);
                    },
                    |(res, (a, b))| {
                        let one = <T as TypeCommon>::Vec::splat(T::ONE);
                        *res = a._eq(b)._mul(one);
                    }
                )
                .collect::<_Tensor<T>>()
        };
        #[cfg(not(feature = "simd"))]
        let ret = {
            self.par_iter()
                .zip(max.par_iter())
                .strided_map(|(a, b)| { a._eq(b)._mul(T::ONE) })
                .collect::<_Tensor<T>>()
        };
        Ok(ret)
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + NormalOut<T, Output = T> + Cmp + tensor_types::into_scalar::IntoScalar<T>,
        bool: NormalOut<T, Output = T>,
        <T as TypeCommon>::Vec: SimdCmp + NormalOut<Output = <T as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp>::Output: NormalOut<
            <T as TypeCommon>::Vec,
            Output = <T as TypeCommon>::Vec
        >
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hardmax(self, axis)?.into()))
    }
}
