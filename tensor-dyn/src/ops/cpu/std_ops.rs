use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use rayon::iter::ParallelIterator;
use tensor_types::convertion::Convertor;
use tensor_types::into_scalar::IntoScalar;
use std::ops::AddAssign;
use std::ops::{
    Add,
    Sub,
    Mul,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Div,
    SubAssign,
    MulAssign,
    RemAssign,
};
use crate::ops::cpu::binary_normal::*;
use tensor_types::dtype::TypeCommon;
use tensor_types::type_promote::BitWiseOut;
use tensor_types::type_promote::FloatOutBinary;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_types::type_promote::NormalOut;
use crate::tensor_base::_Tensor;

macro_rules! normal_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            #[cfg(not(feature = "simd"))]
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape()
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as NormalOut<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    } else if lhs_size < rhs_size {
                        if rhs_size == res_size && U::ID == <T as NormalOut<U>>::Output::ID {
                            let out: _Tensor<U> = rhs.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    } else {
                        if T::ID == <T as NormalOut<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as NormalOut<U>>::Output::ID {
                            let out: _Tensor<U> = rhs.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    }
                }
            }
            #[cfg(feature = "simd")]
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
            where
            T: CommonBounds + NormalOut<U>,
            U: CommonBounds,
            <T as NormalOut<U>>::Output: CommonBounds,
            <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
            T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>
        {
            type Output = _Tensor<<T as NormalOut<U>>::Output>;

            #[cfg_attr(feature = "track_caller", track_caller)]
            fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                let res_shape = predict_broadcast_shape(
                    self.shape(),
                    rhs.shape(),
                ).unwrap();
                let res_size: usize = res_shape.size() as usize;
                let lhs_size: usize = self.layout().real_size();
                let rhs_size: usize = rhs.layout().real_size();
                if lhs_size > rhs_size {
                    if lhs_size == res_size && T::ID == <T as NormalOut<U>>::Output::ID {
                        let out: _Tensor<T> = self.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out_simd(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            |x, y| x.$op3(y),
                            out,
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
                } else if lhs_size < rhs_size {
                    if rhs_size == res_size && U::ID == <T as NormalOut<U>>::Output::ID {
                        let out: _Tensor<U> = rhs.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out_simd(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
                } else {
                    if T::ID == <T as NormalOut<U>>::Output::ID {
                        let out: _Tensor<T> = self.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else if U::ID == <T as NormalOut<U>>::Output::ID {
                        let out: _Tensor<U> = rhs.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
                }
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
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                }
            }
        )*
    };
}

macro_rules! normal_promote_ops_3 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            #[cfg(not(feature = "simd"))]
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                }
            }
            #[cfg(feature = "simd")]
            impl<'a, T, U> $op<&'a _Tensor<U>> for &'a _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>,
                T::Vec: NormalOut<<U as TypeCommon>::Vec, Output = <<T as NormalOut<U>>::Output as TypeCommon>::Vec>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn_simd(&self, &rhs, |x, y| x.$op3(y), |x, y| x.$op3(y)).unwrap();
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
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape()
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let rhs_size: usize = rhs.layout().real_size();
                    if rhs_size == res_size && U::ID == <T as NormalOut<U>>::Output::ID {
                        let out: _Tensor<U> = rhs.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
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

normal_promote_ops_1!([Add, add, _add], [Sub, sub, _sub], [Mul, mul, _mul], [Rem, rem, _rem]);

normal_promote_ops_2!([Add, add, _add], [Sub, sub, _sub], [Mul, mul, _mul], [Rem, rem, _rem]);

normal_promote_ops_3!([Add, add, _add], [Sub, sub, _sub], [Mul, mul, _mul], [Rem, rem, _rem]);

normal_promote_ops_4!([Add, add, _add], [Sub, sub, _sub], [Mul, mul, _mul], [Rem, rem, _rem]);

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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;

                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as BitWiseOut<U>>::Output: IntoScalar<<T as BitWiseOut<U>>::Output>
            {
                type Output = _Tensor<<T as BitWiseOut<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape()
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as FloatOutBinary<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    } else if lhs_size < rhs_size {
                        if rhs_size == res_size && U::ID == <T as FloatOutBinary<U>>::Output::ID {
                            let out: _Tensor<U> = rhs.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    } else {
                        if T::ID == <T as FloatOutBinary<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as FloatOutBinary<U>>::Output::ID {
                            let out: _Tensor<U> = rhs.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else {
                            return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                        }
                    }
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
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape()
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as FloatOutBinary<U>>::Output::ID {
                        let out: _Tensor<T> = self.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
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
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                <T as FloatOutBinary<U>>::Output: IntoScalar<<T as FloatOutBinary<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOutBinary<U>>::Output>;
                #[cfg_attr(feature = "track_caller", track_caller)]
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape()
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let rhs_size: usize = rhs.layout().real_size();
                    if res_size == rhs_size && U::ID == <T as FloatOutBinary<U>>::Output::ID {
                        let out: _Tensor<U> = rhs.clone();
                        let out: Self::Output = out.static_cast().unwrap();
                        return binary_fn_with_out(
                            &self,
                            &rhs,
                            |x, y| x.$op3(y),
                            out
                        ).unwrap();
                    } else {
                        return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
                    }
                }
            }
        )*
    };
}

float_binary_promote_ops_1!([Div, div, _div]);
float_binary_promote_ops_2!([Div, div, _div]);
float_binary_promote_ops_3!([Div, div, _div]);
float_binary_promote_ops_4!([Div, div, _div]);

impl<T, U> PartialEq<_Tensor<U>>
    for _Tensor<T>
    where T: CommonBounds + Convertor, U: CommonBounds + Convertor
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
        $(impl<T> Add<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn add(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._add(y)).unwrap();
            }
        }
        impl<T> Mul<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn mul(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._mul(y)).unwrap();
            }
        }
        impl<T> Sub<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn sub(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._sub(y)).unwrap();
            }
        }
        impl<T> Div<$type> for $($tokens)*_Tensor<T> where T: FloatOutBinary<$type> + CommonBounds, <T as FloatOutBinary<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as FloatOutBinary<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn div(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._div(y)).unwrap();
            }
        }
        impl<T> Rem<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn rem(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._rem(y)).unwrap();
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
        $(impl<T> Add<$($tokens)*_Tensor<T>> for $type where T: CommonBounds, <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T> {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn add(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                return binary_fn(&lhs, &rhs, |x, y| x._add(y)).unwrap();
            }
        }
        impl<T> Mul<$($tokens)*_Tensor<T>> for $type where T: CommonBounds, <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T> {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn mul(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                return binary_fn(&lhs, &rhs, |x, y| x._mul(y)).unwrap();
            }
        }
        impl<T> Sub<$($tokens)*_Tensor<T>> for $type where T: CommonBounds, <$type as NormalOut<T>>::Output: CommonBounds, $type: NormalOut<T> {
            type Output = _Tensor<<$type as NormalOut<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn sub(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                return binary_fn(&lhs, &rhs, |x, y| x._sub(y)).unwrap();
            }
        }
        impl<T> Div<$($tokens)*_Tensor<T>> for $type where T: FloatOutBinary<T> + CommonBounds, <$type as FloatOutBinary<T>>::Output: CommonBounds, $type: FloatOutBinary<T> {
            type Output = _Tensor<<$type as FloatOutBinary<T>>::Output>;
            #[cfg_attr(feature = "track_caller", track_caller)]
            fn div(self, rhs: $($tokens)*_Tensor<T>) -> Self::Output {
                let lhs: _Tensor<$type> = self.into();
                return binary_fn(&lhs, &rhs, |x, y| x._div(y)).unwrap();
            }
        })*
    };
}
use half::f16;
use half::bf16;
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
