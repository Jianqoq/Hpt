use std::ops::{ Add, Neg, Sub };

use tensor_traits::CommonBounds;
use tensor_types::type_promote::NormalOut;
use tensor_types::{ dtype::TypeCommon, type_promote::SimdCmp };
use tensor_types::traits::{ Init, SimdSelect };
use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where
        T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>,
        <T as TypeCommon>::Vec: SimdCmp + NormalOut<Output = <T as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp>::Output: SimdSelect<<T as TypeCommon>::Vec>
{
    #[allow(unused)]
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<_Tensor<T>> {
        #[cfg(feature = "simd")]
        {
            let lambda_vec = <T as TypeCommon>::Vec::splat(lambda);
            let neg_lambda_vec = lambda_vec._neg();
            let bias_vec = <T as TypeCommon>::Vec::splat(bias);

            Ok(
                self
                    .par_iter_simd()
                    .strided_map_simd(
                        |(x, y)| {
                            *x = if y > lambda {
                                y - bias
                            } else if y < -lambda {
                                y + bias
                            } else {
                                T::ZERO
                            };
                        },
                        |(x, y)| {
                            let gt_mask = y._gt(lambda_vec);
                            let lt_mask = y._lt(neg_lambda_vec);
                            let sub_bias = y._sub(bias_vec);
                            let add_bias = y._add(bias_vec);
                            let zero = <T as TypeCommon>::Vec::splat(T::ZERO);
                            let res = gt_mask.select(sub_bias, zero);
                            *x = lt_mask.select(add_bias, res);
                        }
                    )
                    .collect()
            )
        }
        #[cfg(not(feature = "simd"))]
        {
            Ok(
                self
                    .par_iter()
                    .strided_map(|x| {
                        if x > lambda { x - bias } else if x < -lambda { x + bias } else { T::ZERO }
                    })
                    .collect()
            )
        }
    }
}

impl<T> Tensor<T>
    where
        T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>,
        <T as TypeCommon>::Vec: SimdCmp + NormalOut<Output = <T as TypeCommon>::Vec>,
        <<T as TypeCommon>::Vec as SimdCmp>::Output: SimdSelect<<T as TypeCommon>::Vec>
{
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<Tensor<T>> {
        Ok(_Tensor::shrink(self, bias, lambda)?.into())
    }
}
