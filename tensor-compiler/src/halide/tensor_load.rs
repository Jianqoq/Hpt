use std::{ fmt::Display, sync::Arc };
use itertools::izip;

use crate::halide::{ exprs::Load, prime_expr::PrimeExpr, variable::Variable };

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TensorLoad {
    pub(crate) var: Arc<Variable>,
    pub(crate) axes: Arc<Vec<PrimeExpr>>,
    pub(crate) strides: Arc<Vec<PrimeExpr>>,
    pub(crate) hints: Arc<Vec<Hint>>,
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub enum Hint {
    Zero,
    One,
    Unknown,
}

impl TensorLoad {
    pub fn make<
        A: Into<Variable>,
        B: Into<Vec<PrimeExpr>>,
        C: Into<Vec<PrimeExpr>>,
        D: Into<Vec<Hint>>
    >(var: A, axes: B, strides: C, hints: D) -> Self {
        TensorLoad {
            var: Arc::new(var.into()),
            axes: Arc::new(axes.into()),
            strides: Arc::new(strides.into()),
            hints: Arc::new(hints.into()),
        }
    }
}

impl Into<PrimeExpr> for TensorLoad {
    fn into(self) -> PrimeExpr {
        PrimeExpr::TensorLoad(self)
    }
}

impl Into<PrimeExpr> for &TensorLoad {
    fn into(self) -> PrimeExpr {
        PrimeExpr::TensorLoad(self.clone())
    }
}

impl Display for TensorLoad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut indices = PrimeExpr::None;
        for (axes, stride) in izip!(self.axes.iter(), self.strides.iter()) {
            if indices.is_none() {
                indices = axes.clone() * stride.clone();
            } else {
                indices = indices + axes.clone() * stride.clone();
            }
        }
        let load = Load::make(self.var.as_ref(), indices);
        write!(f, "{}", load)
    }
}
