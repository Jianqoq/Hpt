use std::{ fmt::Display, sync::Arc };
use half::{ f16, bf16 };
use itertools::izip;
use tensor_macros::impl_tensor_slice_std_ops;
use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::{ Add, Div, Int, Load, Mul, Rem, Sub },
    prime_expr::PrimeExpr,
    variable::Variable,
};

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TensorSlice {
    pub(crate) var: Arc<Variable>,
    pub(crate) dims: Arc<Vec<PrimeExpr>>,
    pub(crate) strides: Option<Arc<Vec<usize>>>,
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
        if let Some(strides) = &self.strides {
            let load_strides = strides
                .iter()
                .map(|x| { Load::make(format!("{}.strides", self.var.as_ref()), x) });
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
        } else {
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
            strides: None,
        }
    }
    pub fn make_strides<A: Into<Variable>, B: Into<Vec<PrimeExpr>>>(
        var: A,
        dims: B,
        strides: &[usize]
    ) -> Self {
        TensorSlice {
            var: Arc::new(var.into()),
            dims: Arc::new(dims.into()),
            strides: Some(Arc::new(strides.to_vec())),
        }
    }
}

impl std::ops::Add for TensorSlice {
    type Output = PrimeExpr;

    fn add(self, rhs: Self) -> Self::Output {
        Add::make(self, rhs).into()
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct TensorLoad {
    pub(crate) var: Arc<Variable>,
    pub(crate) begins: Arc<Vec<PrimeExpr>>,
    pub(crate) axes: Arc<Vec<PrimeExpr>>,
    pub(crate) steps: Arc<Vec<PrimeExpr>>,
    pub(crate) strides: Arc<Vec<PrimeExpr>>,
    pub(crate) hints: Arc<Vec<Hint>>
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
        D: Into<Vec<PrimeExpr>>,
        E: Into<Vec<PrimeExpr>>,
        F: Into<Vec<Hint>>
    >(var: A, begins: B, axes: C, steps: D, strides: E, hints: F) -> Self {
        TensorLoad {
            var: Arc::new(var.into()),
            begins: Arc::new(begins.into()),
            axes: Arc::new(axes.into()),
            steps: Arc::new(steps.into()),
            strides: Arc::new(strides.into()),
            hints: Arc::new(hints.into())
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
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1i64));
        let zero = PrimeExpr::Int(Int::make(Dtype::I64, 0i64));
        assert!(self.begins.len() == self.axes.len());
        assert!(self.begins.len() == self.steps.len());
        assert!(self.begins.len() == self.strides.len());
        let mut indices = PrimeExpr::None;
        for (begin, axes, step, stride) in izip!(
            self.begins.iter(),
            self.axes.iter(),
            self.steps.iter(),
            self.strides.iter()
        ) {
            if begin == &zero && step == &one {
                if indices.is_none() {
                    indices = axes.clone() * stride.clone();
                } else {
                    indices = indices + axes.clone() * stride.clone();
                }
            } else if step == &one {
                if indices.is_none() {
                    indices = (axes.clone() + begin.clone()) * stride.clone();
                } else {
                    indices = indices + (axes.clone() + begin.clone()) * stride.clone();
                }
            } else {
                if indices.is_none() {
                    indices = (axes.clone() * step.clone() + begin.clone()) * stride.clone();
                } else {
                    indices =
                        indices + (axes.clone() * step.clone() + begin.clone()) * stride.clone();
                }
            }
        }
        let load = Load::make(self.var.as_ref(), indices);
        write!(f, "{}", load)
    }
}

impl_tensor_slice_std_ops!();
