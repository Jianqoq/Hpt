use crate::{tensor::Tensor, tensor_base::_Tensor};
use tensor_traits::CommonBounds;
use tensor_types::{
    convertion::Convertor,
    type_promote::{Cmp, NormalOut, SimdCmp},
};

impl<T> _Tensor<T>
where
    T: CommonBounds
        + NormalOut<T, Output = T>
        + Cmp
        + tensor_types::into_scalar::IntoScalar<T>
        + Convertor,
    bool: NormalOut<T, Output = T>,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: NormalOut<T::Vec, Output = T::Vec>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<_Tensor<T>> {
        let axis = (if axis < 0 {
            (self.layout.ndim() as i64) + axis
        } else {
            axis
        }) as usize;
        let max = self.max(axis as i64, true)?;
        let ret = {
            use tensor_types::traits::Init;
            self.par_iter_simd()
                .zip(max.par_iter_simd())
                .strided_map(
                    |(res, (a, b))| {
                        *res = a._eq(b)._mul(T::ONE);
                    },
                    |(res, (a, b))| {
                        let one = T::Vec::splat(T::ONE);
                        *res = a._eq(b)._mul(one);
                    },
                )
                .collect::<_Tensor<T>>()
        };
        Ok(ret)
    }
}

impl<T> Tensor<T>
where
    T: CommonBounds
        + NormalOut<T, Output = T>
        + Cmp
        + tensor_types::into_scalar::IntoScalar<T>
        + Convertor,
    bool: NormalOut<T, Output = T>,
    T::Vec: SimdCmp + NormalOut<Output = T::Vec>,
    <T::Vec as SimdCmp>::Output: NormalOut<T::Vec, Output = T::Vec>,
{
    #[cfg_attr(feature = "track_caller", track_caller)]
    pub fn hardmax(&self, axis: i64) -> anyhow::Result<Tensor<T>> {
        Ok(Tensor::from(_Tensor::hardmax(self, axis)?.into()))
    }
}
