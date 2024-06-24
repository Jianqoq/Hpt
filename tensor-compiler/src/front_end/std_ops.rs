
macro_rules! impl_std_op {
    (
        $std_op:ident,
        [$($life_time:tt)*],
        $lhs:ident,
        [$($life_time2:tt)*],
        $rhs:ident,
        $op:ident,
        $custom_op:ident
    ) => {
        impl $std_op<$($life_time2)*$rhs> for $($life_time)*$lhs {
            type Output = Tensor;

            fn $op(self, rhs: $($life_time2)*$rhs) -> Self::Output {
                let res_shape = predict_broadcast_shape(&self.shape()[..], &rhs.shape()[..]);
                if let Err(_) = res_shape {
                    todo!();
                }
                let res_shape: Shape = res_shape.unwrap().into();
                let strides = res_shape.to_strides();
                let ret = Tensor::_empty(
                    vec![self.id, rhs.id],
                    self.dtype.$custom_op(rhs.dtype),
                    Op::$std_op,
                    Layout::new(res_shape, strides),
                    None,
                    Rc::new(vec![]),
                    self.ctx.clone()
                );
                ret
            }
        }
    };
}

use std::ops::{Add, Div, Mul, Sub};
use crate::front_end::tensor::Tensor;
use tensor_types::type_promote::NormalOut;
use tensor_common::shape::Shape;
use tensor_common::shape_utils::predict_broadcast_shape;
use tensor_common::layout::Layout;
use tensor_traits::tensor::StaticTensorInfo;
use tensor_types::type_promote::FloatOut;
use crate::op::Op;
use std::rc::Rc;

impl_std_op!(Add, [], Tensor, [], Tensor, add, _add);
impl_std_op!(Sub, [], Tensor, [], Tensor, sub, _sub);
impl_std_op!(Mul, [], Tensor, [], Tensor, mul, _mul);
impl_std_op!(Div, [], Tensor, [], Tensor, div, _div);

impl_std_op!(Add, [], Tensor, [&], Tensor, add, _add);
impl_std_op!(Sub, [], Tensor, [&], Tensor, sub, _sub);
impl_std_op!(Mul, [], Tensor, [&], Tensor, mul, _mul);
impl_std_op!(Div, [], Tensor, [&], Tensor, div, _div);

impl_std_op!(Add, [&], Tensor, [], Tensor, add, _add);
impl_std_op!(Sub, [&], Tensor, [], Tensor, sub, _sub);
impl_std_op!(Mul, [&], Tensor, [], Tensor, mul, _mul);
impl_std_op!(Div, [&], Tensor, [], Tensor, div, _div);

impl_std_op!(Add, [&], Tensor, [&], Tensor, add, _add);
impl_std_op!(Sub, [&], Tensor, [&], Tensor, sub, _sub);
impl_std_op!(Mul, [&], Tensor, [&], Tensor, mul, _mul);
impl_std_op!(Div, [&], Tensor, [&], Tensor, div, _div);