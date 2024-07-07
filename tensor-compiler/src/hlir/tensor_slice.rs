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
        let load_strides = (0..self.dims.len()).map(|x| {
            Load::make(format!("{}.strides", self.var.as_ref()), x)
        });
        let load = Load::make(
            self.var.as_ref(),
            self.dims
                .iter()
                .zip(load_strides)
                .map(|(x, strides)| x.clone() * strides.into())
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
    pub fn dims_(&self) -> &Arc<Vec<PrimeExpr>> {
        &self.dims
    }
    pub fn make<A: Into<Variable>, B: Into<Vec<PrimeExpr>>>(var: A, dims: B) -> Self {
        TensorSlice {
            var: Arc::new(var.into()),
            dims: Arc::new(dims.into()),
        }
    }
}

impl std::ops::Add for TensorSlice {
    type Output = PrimeExpr;

    fn add(self, rhs: Self) -> Self::Output {
        Add::make(self, rhs).into()
    }
}

impl_tensor_slice_std_ops!();
