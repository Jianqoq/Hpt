use tensor_traits::{ CommonBounds, TensorInfo };
use tensor_types::{
    convertion::Convertor,
    into_scalar::IntoScalar,
    type_promote::{ FloatOutBinary, FloatOutUnary, NormalOut },
};

use crate::{ backend::Cpu, tensor_base::_Tensor };

use super::unary::FloatBinaryType;
use tensor_types::dtype::TypeCommon;

impl<T> _Tensor<T, Cpu> {
    pub fn lp_normalization(&self, p: u8, axis: i64) -> anyhow::Result<_Tensor<FloatBinaryType<T>>>
        where
            T: CommonBounds +
                NormalOut<T, Output = T> +
                FloatOutBinary +
                IntoScalar<<T as FloatOutBinary>::Output> +
                Convertor,
            <T as TypeCommon>::Vec: NormalOut<
                <T as TypeCommon>::Vec,
                Output = <T as TypeCommon>::Vec
            >,
            <T as FloatOutBinary>::Output: CommonBounds +
                NormalOut<<T as FloatOutBinary>::Output, Output = <T as FloatOutBinary>::Output> +
                FloatOutUnary<Output = <T as FloatOutBinary>::Output>,
            <<T as FloatOutBinary>::Output as TypeCommon>::Vec: NormalOut<
                <T as TypeCommon>::Vec,
                Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
            > +
                FloatOutUnary<Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec> +
                NormalOut<
                    <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
                    Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
                >,
            T: NormalOut +
                FloatOutBinary<
                    <T as FloatOutBinary>::Output,
                    Output = <T as FloatOutBinary>::Output
                >,
            <T as FloatOutBinary>::Output: NormalOut<T, Output = <T as FloatOutBinary>::Output>,
            <T as TypeCommon>::Vec: NormalOut<
                <<T as FloatOutBinary>::Output as TypeCommon>::Vec,
                Output = <<T as FloatOutBinary>::Output as TypeCommon>::Vec
            > +
                NormalOut<<T as TypeCommon>::Vec, Output = <T as TypeCommon>::Vec>
    {
        let axis = if axis < 0 { (self.shape().len() as i64) + axis } else { axis };
        match p {
            1 => {
                let norm = self.reducel1(axis, false)?;
                Ok(self / norm)
            }
            2 => {
                let norm = self.reducel2(axis, false)?;
                Ok(self / norm)
            }
            _ => {
                unimplemented!();
            }
        }
    }
}
