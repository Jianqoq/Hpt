use std::ops::{ Add, Neg, Sub };

use tensor_traits::CommonBounds;

use crate::{ tensor::Tensor, tensor_base::_Tensor };

impl<T> _Tensor<T>
    where T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>
{
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<_Tensor<T>> {
        // Ok(
        //     self
        //         .par_iter()
        //         .strided_map(|x| {
        //             if x > lambda { x - bias } else if x < -lambda { x + bias } else { T::ZERO }
        //         })
        //         .collect()
        // )
        todo!()
    }
}

impl<T> Tensor<T>
    where T: CommonBounds + PartialOrd + Sub<Output = T> + Neg<Output = T> + Add<Output = T>
{
    pub fn shrink(&self, bias: T, lambda: T) -> anyhow::Result<Tensor<T>> {
        Ok(_Tensor::shrink(self, bias, lambda)?.into())
    }
}
