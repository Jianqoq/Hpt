use rayon::iter::ParallelIterator;
use tensor_traits::tensor::CommonBounds;
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
use tensor_types::type_promote::FloatOut;
use crate::ops::binary_funcs_normal::binary_fn;
use crate::ops::binary_funcs_normal::binary_fn_with_out;
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
                    let res_size: usize = res_shape.size();
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
                    let res_size: usize = res_shape.size();
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
                    let res_size: usize = res_shape.size();
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
                if self.ref_cnt() == 1 && self.parent().is_none() {
                    self.strided_par_iter_mut()
                        .zip(rhs.strided_par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
                } else {
                    panic!("unsafe add_assign operation happened only supports self with ref_cnt == 1")
                }
            }
        })*

        $(impl<'a, T> $op<_Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: _Tensor<T>) {
                if self.ref_cnt() == 1 && self.parent().is_none() {
                    self.strided_par_iter_mut()
                        .zip(rhs.strided_par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
                } else {
                    panic!("unsafe add_assign operation happened only supports self with ref_cnt == 1")
                }
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for &'a _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                if self.ref_cnt() == 1 && self.parent().is_none() {
                    self.strided_par_iter_mut()
                        .zip(rhs.strided_par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
                } else {
                    panic!("unsafe add_assign operation happened only supports self with ref_cnt == 1")
                }
            }
        })*

        $(impl<'a, T> $op<&'a _Tensor<T>> for _Tensor<T> where T: CommonBounds + NormalOut<Output=T> {
            fn $op2(&mut self, rhs: &'a _Tensor<T>) {
                if self.ref_cnt() == 1 && self.parent().is_none() {
                    self.strided_par_iter_mut()
                        .zip(rhs.strided_par_iter())
                        .for_each(|(x, y)| {
                            *x = x.$op3(y);
                        });
                } else {
                    panic!("unsafe add_assign operation happened only supports self with ref_cnt == 1")
                }
            }
        })*
    };
}

normal_promote_ops_1!([Add, add, __add], [Sub, sub, __sub], [Mul, mul, __mul], [Rem, rem, __rem]);

normal_promote_ops_2!([Add, add, __add], [Sub, sub, __sub], [Mul, mul, __mul], [Rem, rem, __rem]);

normal_promote_ops_3!([Add, add, __add], [Sub, sub, __sub], [Mul, mul, __mul], [Rem, rem, __rem]);

normal_promote_ops_4!([Add, add, __add], [Sub, sub, __sub], [Mul, mul, __mul], [Rem, rem, __rem]);

normal_promote_ops_assign!(
    [AddAssign, add_assign, __add],
    [SubAssign, sub_assign, __sub],
    [MulAssign, mul_assign, __mul],
    [RemAssign, rem_assign, __rem]
);

macro_rules! bitwise_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size();
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as Bitwise<U>>::Output::ID {
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
                        if rhs_size == res_size && U::ID == <T as Bitwise<U>>::Output::ID {
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
                        if T::ID == <T as Bitwise<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as Bitwise<U>>::Output::ID {
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = shape_to_size(&res_shape);
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as Bitwise<U>>::Output::ID {
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = shape_to_size(&res_shape);
                    let rhs_size: usize = rhs.layout().real_size();
                    if rhs_size == res_size && U::ID == <T as Bitwise<U>>::Output::ID {
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

bitwise_promote_ops_1!([BitAnd, bitand, __and], [BitOr, bitor, __or], [BitXor, bitxor, __xor]);

bitwise_promote_ops_2!([BitAnd, bitand, __and], [BitOr, bitor, __or], [BitXor, bitxor, __xor]);

bitwise_promote_ops_3!([BitAnd, bitand, __and], [BitOr, bitor, __or], [BitXor, bitxor, __xor]);

bitwise_promote_ops_4!([BitAnd, bitand, __and], [BitOr, bitor, __or], [BitXor, bitxor, __xor]);

macro_rules! shift_promote_ops_1 {
    ($([$op:ident, $op2:ident, $op3:ident]),*) => {
        $(
            impl<T, U> $op<_Tensor<U>> for _Tensor<T>
                where
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = shape_to_size(&res_shape);
                    let lhs_size: usize = self.layout().real_size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if lhs_size > rhs_size {
                        if lhs_size == res_size && T::ID == <T as Bitwise<U>>::Output::ID {
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
                        if rhs_size == res_size && U::ID == <T as Bitwise<U>>::Output::ID {
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
                        if T::ID == <T as Bitwise<U>>::Output::ID {
                            let out: _Tensor<T> = self.clone();
                            let out: Self::Output = out.static_cast().unwrap();
                            return binary_fn_with_out(
                                &self,
                                &rhs,
                                |x, y| x.$op3(y),
                                out
                            ).unwrap();
                        } else if U::ID == <T as Bitwise<U>>::Output::ID {
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: &'a _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size();
                    let lhs_size: usize = self.layout().real_size();
                    if lhs_size == res_size && T::ID == <T as Bitwise<U>>::Output::ID {
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
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
                T: CommonBounds + Bitwise<U>,
                U: CommonBounds,
                <T as Bitwise<U>>::Output: CommonBounds,
                <T as Bitwise<U>>::Output: IntoScalar<<T as Bitwise<U>>::Output>
            {
                type Output = _Tensor<<T as Bitwise<U>>::Output>;
                fn $op2(self, rhs: _Tensor<U>) -> Self::Output {
                    let res_shape = predict_broadcast_shape(
                        self.shape(),
                        rhs.shape(),
                    ).unwrap();
                    let res_size: usize = res_shape.size();
                    let rhs_size: usize = rhs.layout().real_size();
                    if res_size == rhs_size && U::ID == <T as Bitwise<U>>::Output::ID {
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

shift_promote_ops_1!([Shl, shl, __shl], [Shr, shr, __shr]);
shift_promote_ops_2!([Shl, shl, __shl], [Shr, shr, __shr]);
shift_promote_ops_3!([Shl, shl, __shl], [Shr, shr, __shr]);
shift_promote_ops_4!([Shl, shl, __shl], [Shr, shr, __shr]);

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
                    let res_size: usize = res_shape.size();
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
                    let res_size: usize = res_shape.size();
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
                    let res_size: usize = res_shape.size();
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

float_binary_promote_ops_1!([Div, div, __div]);
float_binary_promote_ops_2!([Div, div, __div]);
float_binary_promote_ops_3!([Div, div, __div]);
float_binary_promote_ops_4!([Div, div, __div]);

impl<T, U> PartialEq<_Tensor<U>>
    for _Tensor<T>
    where T: CommonBounds + BinaryBool<U>, U: CommonBounds
{
    fn eq(&self, other: &_Tensor<U>) -> bool {
        if self.size != other.size {
            return false;
        }
        if self.shape() != other.shape() {
            return false;
        }
        if self.shape().len() == 0 {
            return self.as_raw()[0].__eq(other.as_raw()[0]);
        }
        let folder = self
            .strided_par_iter()
            .zip(other.strided_par_iter())
            .fold(
                || true,
                |acc, (x, y)| if x.__eq(y) { acc && true } else { acc && false }
            );
        return folder.reduce(
            || true,
            |acc, x| acc && x
        );
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
                return binary_fn(&self, &rhs, |x, y| x.__add(y)).unwrap();
            }
        }
        impl<T> Mul<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn mul(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x.__mul(y)).unwrap();
            }
        }
        impl<T> Sub<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn sub(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x.__sub(y)).unwrap();
            }
        }
        impl<T> Div<$type> for $($tokens)*_Tensor<T> where T: FloatOut<$type> + CommonBounds, <T as FloatOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as FloatOut<$type>>::Output>;
            fn div(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x.__div(y)).unwrap();
            }
        }
        impl<T> Rem<$type> for $($tokens)*_Tensor<T> where T: NormalOut<$type> + CommonBounds, <T as NormalOut<$type>>::Output: CommonBounds {
            type Output = _Tensor<<T as NormalOut<$type>>::Output>;
            fn rem(self, rhs: $type) -> Self::Output {
                let rhs: _Tensor<$type> = rhs.into();
                return binary_fn(&self, &rhs, |x, y| x.__rem(y)).unwrap();
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
