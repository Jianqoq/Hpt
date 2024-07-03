use std::sync::Arc;
use half::{ f16, bf16 };
use tensor_macros::impl_tensor_slice_std_ops;

use crate::halide::{ exprs::{ Add, Div, Load, Mod, Mul, Sub }, prime_expr::PrimeExpr };
use super::tensor::Tensor;

pub struct TensorSlice {
    pub(crate) tensor: Arc<Tensor>,
    pub(crate) indices: Arc<Vec<PrimeExpr>>,
}

impl Into<PrimeExpr> for TensorSlice {
    fn into(self) -> PrimeExpr {
        Load::make(
            self.tensor.name(),
            self.indices
                .iter()
                .map(|e| e.clone())
                .reduce(|a, b| a + b)
                .unwrap()
        ).into()
    }
}

impl Into<PrimeExpr> for &TensorSlice {
    fn into(self) -> PrimeExpr {
        Load::make(
            self.tensor.name(),
            self.indices
                .iter()
                .map(|e| e.clone())
                .reduce(|a, b| a + b)
                .unwrap()
        ).into()
    }
}

impl_tensor_slice_std_ops!();
