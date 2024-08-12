use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{ into_scalar::IntoScalar, type_promote::{ FloatOut, NormalOut } };

use crate::{ backend::Cpu, tensor_base::_Tensor };

use super::{ reduce::{ reduce, reduce3 }, unary::FloatType };
use tensor_types::dtype::TypeCommon;

impl<T> _Tensor<T, Cpu>
    where
        T: CommonBounds +
            NormalOut<Output = T> +
            FloatOut +
            FloatOut<FloatType<T>, Output = FloatType<T>>,
        FloatType<T>: CommonBounds +
            IntoScalar<FloatType<T>> +
            FloatOut<FloatType<T>, Output = FloatType<T>> +
            NormalOut<<T as NormalOut>::Output, Output = FloatType<T>> +
            NormalOut<FloatType<T>, Output = FloatType<T>> +
            PartialEq
{
    pub fn lp_normalization(&self, p: u8, axis: i64) -> anyhow::Result<_Tensor<FloatType<T>>> {
        let axis = if axis < 0 { (self.shape().len() as i64) + axis } else { axis };
        match p {
            1 => {
                let norm = reduce(
                    self,
                    |a, b| a._add(b._abs()),
                    &[axis as usize],
                    T::ZERO,
                    true,
                    false,
                    None
                )?;
                Ok(self / norm)
            }
            2 => {
                let norm = reduce3(
                    self,
                    |a: <T as FloatOut>::Output, b| a._add(<T as NormalOut>::_square(b)),
                    |a, b| a._add(b),
                    move |a| {
                        let res = a._sqrt();
                        if res == <T as FloatOut>::Output::ZERO {
                            <T as FloatOut>::Output::ONE
                        } else {
                            res
                        }
                    },
                    &[axis as usize],
                    <T as FloatOut>::Output::ZERO,
                    true,
                    false,
                    None
                )?;
                Ok(self / norm)
            }
            _ => { unimplemented!() }
        }
    }
}
