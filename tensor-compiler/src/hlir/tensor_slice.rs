use std::{ fmt::Display, sync::Arc };
use half::{ f16, bf16 };
use tensor_macros::impl_tensor_slice_std_ops;

use crate::halide::{
    exprs::{ Add, Div, Load, Mod, Mul, Sub },
    prime_expr::PrimeExpr,
    variable::Variable,
};

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TensorSlice {
    pub(crate) var: Arc<Variable>,
    pub(crate) dims: Arc<Vec<PrimeExpr>>,
    pub(crate) strides: Arc<Vec<PrimeExpr>>,
}

impl Into<PrimeExpr> for TensorSlice {
    fn into(self) -> PrimeExpr {
        PrimeExpr::TensorSlice(self)
    }
}

impl Into<PrimeExpr> for &TensorSlice {
    fn into(self) -> PrimeExpr {
        PrimeExpr::TensorSlice(self.clone())
    }
}

impl Display for TensorSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let load = Load::make(
            self.var.as_ref(),
            self.dims
                .iter()
                .zip(self.strides.iter())
                .map(|(x, y)| x.clone() * y.clone())
                .reduce(|x, y| x + y)
                .unwrap()
        );
        write!(f, "{}", load)
    }
}

impl TensorSlice {
    pub fn name(&self) -> &Variable {
        &self.var
    }
    pub fn dims(&self) -> &[PrimeExpr] {
        &self.dims
    }
    pub fn strides(&self) -> &[PrimeExpr] {
        &self.strides
    }
    pub fn make<A: Into<Variable>, B: Into<Vec<PrimeExpr>>, C: Into<Vec<PrimeExpr>>>(
        var: A,
        dims: B,
        strides: C
    ) -> Self {
        TensorSlice {
            var: Arc::new(var.into()),
            dims: Arc::new(dims.into()),
            strides: Arc::new(strides.into()),
        }
    }
}

impl_tensor_slice_std_ops!();
