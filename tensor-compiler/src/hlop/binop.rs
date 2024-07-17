use std::sync::Arc;
use paste::paste;
use tensor_types::type_promote::{ NormalOut, FloatOut, BitWiseOut };

use crate::{ halide::{ exprs::*, prime_expr::PrimeExpr }, hlir::tensor::{ Tensor, _compute } };

use super::broadcast::predict_brocast_shape;

macro_rules! impl_std_op {
    (
        [$($lhs:tt)*],
        [$($rhs:tt)*],
        $op_name:ident,
        $op:ident
    ) => {
        paste!{
            impl std::ops::$op<$($rhs)*> for $($lhs)* {
                type Output = Tensor;
            
                fn $op_name(self, rhs: $($rhs)*) -> Self::Output {
                    let lhs_shape = &self.shape;
                    let rhs_shape = &rhs.shape;
                    let (res_shape, lhs_indices, rhs_indices) = predict_brocast_shape(lhs_shape, rhs_shape);
                    let lhs_indices = Arc::new(lhs_indices);
                    let rhs_indices = Arc::new(rhs_indices);
                    _compute(
                        self.dtype.[<_ $op_name>](rhs.dtype),
                        res_shape,
                        vec![self.clone(), rhs.clone()],
                        format!("{}_{}_{}", self.name, stringify!($op_name), rhs.name),
                        move |inputs, indices| {
                            let lhs_indices = lhs_indices.clone();
                            let rhs_indices = rhs_indices.clone();
                            let lhs_indices = lhs_indices
                                .iter()
                                .map(|x| indices[*x].var().clone().into())
                                .collect::<Vec<PrimeExpr>>();
                            let rhs_indices = rhs_indices
                                .iter()
                                .map(|x| indices[*x].var().clone().into())
                                .collect::<Vec<PrimeExpr>>();
                            $op::make(inputs[0]._slice(&lhs_indices), inputs[1]._slice(&rhs_indices)).into()
                        }
                    )
                }
            }
        }
    };
}

impl_std_op!([Tensor], [Tensor], add, Add);
impl_std_op!([&Tensor], [Tensor], add, Add);
impl_std_op!([Tensor], [&Tensor], add, Add);
impl_std_op!([&Tensor], [&Tensor], add, Add);

impl_std_op!([Tensor], [Tensor], sub, Sub);
impl_std_op!([&Tensor], [Tensor], sub, Sub);
impl_std_op!([Tensor], [&Tensor], sub, Sub);
impl_std_op!([&Tensor], [&Tensor], sub, Sub);

impl_std_op!([Tensor], [Tensor], mul, Mul);
impl_std_op!([&Tensor], [Tensor], mul, Mul);
impl_std_op!([Tensor], [&Tensor], mul, Mul);
impl_std_op!([&Tensor], [&Tensor], mul, Mul);

impl_std_op!([Tensor], [Tensor], div, Div);
impl_std_op!([&Tensor], [Tensor], div, Div);
impl_std_op!([Tensor], [&Tensor], div, Div);
impl_std_op!([&Tensor], [&Tensor], div, Div);

impl_std_op!([Tensor], [Tensor], rem, Rem);
impl_std_op!([&Tensor], [Tensor], rem, Rem);
impl_std_op!([Tensor], [&Tensor], rem, Rem);
impl_std_op!([&Tensor], [&Tensor], rem, Rem);

impl_std_op!([Tensor], [Tensor], bitand, BitAnd);
impl_std_op!([&Tensor], [Tensor], bitand, BitAnd);
impl_std_op!([Tensor], [&Tensor], bitand, BitAnd);
impl_std_op!([&Tensor], [&Tensor], bitand, BitAnd);

impl_std_op!([Tensor], [Tensor], bitor, BitOr);
impl_std_op!([&Tensor], [Tensor], bitor, BitOr);
impl_std_op!([Tensor], [&Tensor], bitor, BitOr);
impl_std_op!([&Tensor], [&Tensor], bitor, BitOr);

impl_std_op!([Tensor], [Tensor], bitxor, BitXor);
impl_std_op!([&Tensor], [Tensor], bitxor, BitXor);
impl_std_op!([Tensor], [&Tensor], bitxor, BitXor);
impl_std_op!([&Tensor], [&Tensor], bitxor, BitXor);

impl_std_op!([Tensor], [Tensor], shl, Shl);
impl_std_op!([&Tensor], [Tensor], shl, Shl);
impl_std_op!([Tensor], [&Tensor], shl, Shl);
impl_std_op!([&Tensor], [&Tensor], shl, Shl);

impl_std_op!([Tensor], [Tensor], shr, Shr);
impl_std_op!([&Tensor], [Tensor], shr, Shr);
impl_std_op!([Tensor], [&Tensor], shr, Shr);
impl_std_op!([&Tensor], [&Tensor], shr, Shr);
