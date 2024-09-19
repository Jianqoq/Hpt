use crate::ops::cpu::binary_normal::*;
use crate::tensor_base::_Tensor;
use rayon::iter::ParallelIterator;
use std::ops::AddAssign;
use std::ops::{
    Add, BitAnd, BitOr, BitXor, Div, Mul, MulAssign, Rem, RemAssign, Shl, Shr, Sub, SubAssign,
};
use tensor_traits::tensor::{CommonBounds, TensorInfo};
use tensor_types::convertion::Convertor;
use tensor_types::dtype::TypeCommon;
use tensor_types::into_scalar::IntoScalar;
use tensor_types::type_promote::BitWiseOut;
use tensor_types::type_promote::FloatOutBinary;
use tensor_types::type_promote::NormalOut;

macro_rules! normal_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
            where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
        {
            type Output = _Tensor<<T as NormalOut<U>>::Output>;

            #[cfg_attr(feature = "track_caller", track_caller)]
            fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                return binary_fn_with_out_simd(
                    &self,
                    &rhs,
                    |x, y| x.$op3(y),
                    |x, y| x.$op3(y),
                    None::<_Tensor<<T as NormalOut<U>>::Output>>,
                ).unwrap();
            }
        }
        )*
    };
}

macro_rules! normal_promote_ops_2 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
                T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as NormalOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! normal_promote_ops_3 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
                T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as NormalOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! normal_promote_ops_4 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<_Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
                T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as NormalOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! normal_promote_ops_assign {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(impl<T> $op<_Tensor<T>> for _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: _Tensor<T>) {
                    self.par_iter_mut()
                        .zip(rhs.par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<_Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: _Tensor<T>) {
                    self.par_iter_mut()
                        .zip(rhs.par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                    self.par_iter_mut()
                        .zip(rhs.par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                    self.par_iter_mut()
                        .zip(rhs.par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*
    };
}

normal_promote_ops_1!(
    [Add, add, _add],
    [Sub, sub, _sub],
    [Mul, mul, _mul],
    [Rem, rem, _rem]
);

normal_promote_ops_2!(
    [Add, add, _add],
    [Sub, sub, _sub],
    [Mul, mul, _mul],
    [Rem, rem, _rem]
);

normal_promote_ops_3!(
    [Add, add, _add],
    [Sub, sub, _sub],
    [Mul, mul, _mul],
    [Rem, rem, _rem]
);

normal_promote_ops_4!(
    [Add, add, _add],
    [Sub, sub, _sub],
    [Mul, mul, _mul],
    [Rem, rem, _rem]
);

normal_promote_ops_assign!(
    [AddAssign, add_assign, _add],
    [SubAssign, sub_assign, _sub],
    [MulAssign, mul_assign, _mul],
    [RemAssign, rem_assign, _rem]
);

macro_rules! bitwise_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! bitwise_promote_ops_2 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! bitwise_promote_ops_3 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! bitwise_promote_ops_4 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<_Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

bitwise_promote_ops_1!(
    [BitAnd, bitand, _bitand],
    [BitOr, bitor, _bitor],
    [BitXor, bitxor, _bitxor]
);

bitwise_promote_ops_2!(
    [BitAnd, bitand, _bitand],
    [BitOr, bitor, _bitor],
    [BitXor, bitxor, _bitxor]
);

bitwise_promote_ops_3!(
    [BitAnd, bitand, _bitand],
    [BitOr, bitor, _bitor],
    [BitXor, bitxor, _bitxor]
);

bitwise_promote_ops_4!(
    [BitAnd, bitand, _bitand],
    [BitOr, bitor, _bitor],
    [BitXor, bitxor, _bitxor]
);

macro_rules! shift_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! shift_promote_ops_2 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! shift_promote_ops_3 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! shift_promote_ops_4 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<_Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + BitWiseOut<U>,
                U: CommonBounds,
                <T as BitWiseOut<U>>::Output: CommonBounds,
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>,
                T::Vec: BitWiseOut<U::Vec, Output = <<T as BitWiseOut<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as BitWiseOut<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

shift_promote_ops_1!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_2!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_3!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_4!([Shl, shl, _shl], [Shr, shr, _shr]);

macro_rules! float_binary_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + FloatOutBinary<U>,
                U: CommonBounds,
                <T as FloatOutBinary<U>>::Output: CommonBounds,
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
                T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as FloatOutBinary<U>>::Output>>,
                    ).unwrap();
                }
            }
        )*
    };
}

macro_rules! float_binary_promote_ops_2 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + FloatOutBinary<U>,
                U: CommonBounds,
                <T as FloatOutBinary<U>>::Output: CommonBounds,
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
                T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as FloatOutBinary<U>>::Output>>,
                    ).unwrap()
                }
            }
        )*
    };
}

macro_rules! float_binary_promote_ops_3 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + FloatOutBinary<U>,
                U: CommonBounds,
                <T as FloatOutBinary<U>>::Output: CommonBounds,
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
                T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as FloatOutBinary<U>>::Output>>,
                    ).unwrap()
                }
            }
        )*
    };
}

macro_rules! float_binary_promote_ops_4 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<'a, T, U> $op<_Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + FloatOutBinary<U>,
                U: CommonBounds,
                <T as FloatOutBinary<U>>::Output: CommonBounds,
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>,
                T::Vec: FloatOutBinary<U::Vec, Output = <<T as FloatOutBinary<U>>::Output as TypeCommon>::Vec>,
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    binary_fn_with_out_simd(
                        &self,
                        &rhs,
                        |x, y| x.$op3(y),
                        |x, y| x.$op3(y),
                        None::<_Tensor<<T as FloatOutBinary<U>>::Output>>,
                    ).unwrap()
                }
            }
        )*
    };
}

float_binary_promote_ops_1!([Div, div, _div]);
float_binary_promote_ops_2!([Div, div, _div]);
float_binary_promote_ops_3!([Div, div, _div]);
float_binary_promote_ops_4!([Div, div, _div]);

impl<T, U> PartialEq<_Tensor<U>> for _Tensor<T>
where
    T: CommonBounds + Convertor,
    U: CommonBounds + Convertor,
{
    fn eq(&self, other: &_Tensor<U>) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        self.allclose(other)
    }
}

macro_rules! normal_scalar_rhs {
    (
        $([
            $type:ident,
            [$($tokens:tt)*]
        ]),*
    ) => {
        $(impl<T> Add<$type> for $($tokens)*_Tensor<T>
         where T: NormalOut<$type> + CommonBounds,
         <T as NormalOut<$type>>::Output: CommonBounds,
         T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn add(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                binary_fn_with_out_simd(&self, &rhs, |x, y| x._add(y), |x, y| x._add(y), None::<_Tensor<<T as NormalOut<$type>>::Output>>).unwrap()
            }
        }
        impl<T> Mul<$type> for $($tokens)*_Tensor<T>
        where T: NormalOut<$type> + CommonBounds,
        <T as NormalOut<$type>>::Output: CommonBounds,
        T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn mul(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                binary_fn_with_out_simd(&self, &rhs, |x, y| x._mul(y), |x, y| x._mul(y), None::<_Tensor<<T as NormalOut<$type>>::Output>>).unwrap()
            }
        }
        impl<T> Sub<$type> for $($tokens)*_Tensor<T>
        where T: NormalOut<$type> + CommonBounds,
        <T as NormalOut<$type>>::Output: CommonBounds,
        T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn sub(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                binary_fn_with_out_simd(&self, &rhs, |x, y| x._sub(y), |x, y| x._sub(y), None::<_Tensor<<T as NormalOut<$type>>::Output>>).unwrap()
            }
        }
        impl<T> Div<$type> for $($tokens)*_Tensor<T>
        where T: FloatOutBinary<$type> + CommonBounds,
        <T as FloatOutBinary<$type>>::Output: CommonBounds,
        T::Vec: FloatOutBinary<<$type as TypeCommon>::Vec, Output = <<T as FloatOutBinary<$type>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<T as FloatOutBinary<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn div(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                binary_fn_with_out_simd(&self, &rhs, |x, y| x._div(y), |x, y| x._div(y), None::<_Tensor<<T as FloatOutBinary<$type>>::Output>>).unwrap()
            }
        }
        impl<T> Rem<$type> for $($tokens)*_Tensor<T>
        where T: NormalOut<$type> + CommonBounds,
        <T as NormalOut<$type>>::Output: CommonBounds,
        T::Vec: NormalOut<<$type as TypeCommon>::Vec, Output = <<T as NormalOut<$type>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn rem(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                binary_fn_with_out_simd(&self, &rhs, |x, y| x._rem(y), |x, y| x._rem(y), None::<_Tensor<<T as NormalOut<$type>>::Output>>).unwrap()
            }
        })*
    };
}

macro_rules! normal_scalar_lhs {
    (
        $([
            $type:ident,
            [$($tokens:tt)*]
        ]),*
    ) => {
        $(impl<T> Add<$($tokens)*_Tensor<T>> for $type
        where T: CommonBounds,
        <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
        <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn add(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._add(y), |x, y| x._add(y), None::<_Tensor<<$type as NormalOut<T>>::Output>>).unwrap()
            }
        }
        impl<T> Mul<$($tokens)*_Tensor<T>> for $type
        where T: CommonBounds,
        <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
        <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn mul(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._mul(y), |x, y| x._mul(y), None::<_Tensor<<$type as NormalOut<T>>::Output>>).unwrap()
            }
        }
        impl<T> Sub<$($tokens)*_Tensor<T>> for $type 
        where T: CommonBounds, 
        <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T>,
        <$type as TypeCommon>::Vec: NormalOut<<T as TypeCommon>::Vec, Output = <<$type as NormalOut<T>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn sub(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._sub(y), |x, y| x._sub(y), None::<_Tensor<<$type as NormalOut<T>>::Output>>).unwrap()
            }
        }
        impl<T> Div<$($tokens)*_Tensor<T>> for $type 
        where T: FloatOutBinary<T> + CommonBounds, 
        <$type as FloatOutBinary<T>>::Output: CommonBounds, $type: FloatOutBinary<T>,
        <$type as TypeCommon>::Vec: FloatOutBinary<<T as TypeCommon>::Vec, Output = <<$type as FloatOutBinary<T>>::Output as TypeCommon>::Vec>,
         {
            type Output = _Tensor<<$type as FloatOutBinary<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn div(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                binary_fn_with_out_simd(&lhs, &rhs, |x, y| x._div(y), |x, y| x._div(y), None::<_Tensor<<$type as FloatOutBinary<T>>::Output>>).unwrap()
            }
        })*
    };
}
use half::bf16;
use half::f16;
use num::complex::Complex32;
use num::complex::Complex64;
normal_scalar_rhs!(
    [bool, []],
    [i8, []],
    [i16, []],
    [i32, []],
    [i64, []],
    [u8, []],
    [u16, []],
    [u32, []],
    [u64, []],
    [f16, []],
    [f32, []],
    [f64, []],
    [bf16, []],
    [Complex32, []],
    [Complex64, []]
);

normal_scalar_rhs!(
    [bool, [&]],
    [i8, [&]],
    [i16, [&]],
    [i32, [&]],
    [i64, [&]],
    [u8, [&]],
    [u16, [&]],
    [u32, [&]],
    [u64, [&]],
    [f16, [&]],
    [f32, [&]],
    [f64, [&]],
    [bf16, [&]],
    [Complex32, [&]],
    [Complex64, [&]]
);

normal_scalar_lhs!(
    [bool, []],
    [i8, []],
    [i16, []],
    [i32, []],
    [i64, []],
    [u8, []],
    [u16, []],
    [u32, []],
    [u64, []],
    [f16, []],
    [f32, []],
    [f64, []],
    [bf16, []],
    [Complex32, []],
    [Complex64, []]
);

normal_scalar_lhs!(
    [bool, [&]],
    [i8, [&]],
    [i16, [&]],
    [i32, [&]],
    [i64, [&]],
    [u8, [&]],
    [u16, [&]],
    [u32, [&]],
    [u64, [&]],
    [f16, [&]],
    [f32, [&]],
    [f64, [&]],
    [bf16, [&]],
    [Complex32, [&]],
    [Complex64, [&]]
);
