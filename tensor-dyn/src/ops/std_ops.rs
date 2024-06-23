use tensor_traits::tensor::{ CommonBounds, TensorInfo };
use rayon::iter::ParallelIterator;
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
use tensor_types::dtype::TypeCommon;
use tensor_types::type_promote::BitWiseOut;
use tensor_types::type_promote::FloatOut;
use crate::ops::binary_normal::binary_fn;
use crate::ops::binary_normal::binary_fn_with_out;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_types::type_promote::NormalOut;
use crate::tensor::_Tensor;

macro_rules! normal_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + NormalOut<U>,
                U: CommonBounds,
                <T as NormalOut<U>>::Output: CommonBounds,
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
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
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
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
                <T as NormalOut<U>>::Output: IntoScalar<<T as NormalOut<U>>::Output>
            {
                type Output = _Tensor<<T as NormalOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    return binary_fn(&self, &rhs, |x, y| x.$op3(y)).unwrap();
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
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<_Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: _Tensor<T>) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                    self.iter_mut()
                        .zip(rhs.iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                    self.iter_mut()
                        .zip(rhs.iter())
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
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as BitWiseOut<U>>::Output::ID {
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
                        if rhs_size == res_size && U::ID == <T as BitWiseOut<U>>::Output::ID {
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
                        if T::ID == <T as BitWiseOut<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as BitWiseOut<U>>::Output::ID {
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
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as BitWiseOut<U>>::Output::ID {
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
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let rhs_size: usize = rhs.layout().real_size();
                    if rhs_size == res_size && U::ID == <T as BitWiseOut<U>>::Output::ID {
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

bitwise_promote_ops_1!([BitAnd, bitand, _and], [BitOr, bitor, _or], [BitXor, bitxor, _xor]);

bitwise_promote_ops_2!([BitAnd, bitand, _and], [BitOr, bitor, _or], [BitXor, bitxor, _xor]);

bitwise_promote_ops_3!([BitAnd, bitand, _and], [BitOr, bitor, _or], [BitXor, bitxor, _xor]);

bitwise_promote_ops_4!([BitAnd, bitand, _and], [BitOr, bitor, _or], [BitXor, bitxor, _xor]);

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
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as BitWiseOut<U>>::Output::ID {
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
                        if rhs_size == res_size && U::ID == <T as BitWiseOut<U>>::Output::ID {
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
                        if T::ID == <T as BitWiseOut<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as BitWiseOut<U>>::Output::ID {
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
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as BitWiseOut<U>>::Output::ID {
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
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let rhs_size: usize = rhs.layout().real_size();
                    if res_size == rhs_size && U::ID == <T as BitWiseOut<U>>::Output::ID {
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

shift_promote_ops_1!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_2!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_3!([Shl, shl, _shl], [Shr, shr, _shr]);
shift_promote_ops_4!([Shl, shl, _shl], [Shr, shr, _shr]);

macro_rules! float_binary_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + FloatOut<U>,
                U: CommonBounds,
                <T as FloatOut<U>>::Output: CommonBounds,
                <T as FloatOut<U>>::Output: IntoScalar<<T as FloatOut<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOut<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as FloatOut<U>>::Output::ID {
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
                        if rhs_size == res_size && U::ID == <T as FloatOut<U>>::Output::ID {
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
                        if T::ID == <T as FloatOut<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as FloatOut<U>>::Output::ID {
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
                T: CommonBounds + FloatOut<U>,
                U: CommonBounds,
                <T as FloatOut<U>>::Output: CommonBounds,
                <T as FloatOut<U>>::Output: IntoScalar<<T as FloatOut<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOut<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as FloatOut<U>>::Output::ID {
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
                T: CommonBounds + FloatOut<U>,
                U: CommonBounds,
                <T as FloatOut<U>>::Output: CommonBounds,
                <T as FloatOut<U>>::Output: IntoScalar<<T as FloatOut<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOut<U>>::Output>;
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
                T: CommonBounds + FloatOut<U>,
                U: CommonBounds,
                <T as FloatOut<U>>::Output: CommonBounds,
                <T as FloatOut<U>>::Output: IntoScalar<<T as FloatOut<U>>::Output>
            {
                type Output = _Tensor<<T as FloatOut<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size() as usize;
                    let rhs_size: usize = rhs.layout().real_size();
                    if res_size == rhs_size && U::ID == <T as FloatOut<U>>::Output::ID {
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

impl<T, U> PartialEq<_Tensor<U>> for _Tensor<T> where T: CommonBounds, U: CommonBounds {
    fn eq(&self, other: &_Tensor<U>) -> bool {
        if self.size() != other.size() {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        (self.ptr().ptr as usize) == (other.ptr().ptr as usize)
    }
}

macro_rules! normal_scalar {
    (
        $([
            $type:ident,
            [$($tokens:tt)*]
        ]),*
    ) => {
        $(impl<T> Add<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn add(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._add(y)).unwrap();
            }
        }
        impl<T> Mul<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn mul(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._mul(y)).unwrap();
            }
        }
        impl<T> Sub<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn sub(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._sub(y)).unwrap();
            }
        }
        impl<T> Div<$type> for $($tokens)*_Tensor<T> where T: FloatOut<$type> + CommonBounds, <T as FloatOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as FloatOut<$type>>::Output>;
            fn div(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._div(y)).unwrap();
            }
        }
        impl<T> Rem<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn rem(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x._rem(y)).unwrap();
            }
        })*
    };
}
use half::f16;
use half::bf16;
use num::complex::Complex32;
use num::complex::Complex64;
normal_scalar!(
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

normal_scalar!(
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
