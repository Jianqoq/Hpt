use std::sync::Arc;

use colored::Colorize;

use itertools::izip;
use tensor_types::dtype::Dtype;

use crate::halide::traits::{ Accepter, AccepterMut };

use super::{
    exprs::Int,
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, IRMutVisitor, IRVisitor },
    variable::Variable,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct StoreStmt {
    pub(crate) var: Variable,
    pub(crate) begins: Arc<Vec<PrimeExpr>>,
    pub(crate) steps: Arc<Vec<PrimeExpr>>,
    pub(crate) axes: Arc<Vec<PrimeExpr>>,
    pub(crate) strides: Arc<Vec<PrimeExpr>>,
    pub(crate) val: Arc<PrimeExpr>,
}

impl StoreStmt {
    pub fn var(&self) -> &Variable {
        &self.var
    }

    pub fn begins(&self) -> &Vec<PrimeExpr> {
        &self.begins
    }

    pub fn steps(&self) -> &Vec<PrimeExpr> {
        &self.steps
    }

    pub fn axes(&self) -> &Vec<PrimeExpr> {
        &self.axes
    }

    pub fn strides(&self) -> &Vec<PrimeExpr> {
        &self.strides
    }

    pub fn val(&self) -> &PrimeExpr {
        &self.val
    }

    pub fn make<
        A: Into<Variable>,
        B: Into<Vec<PrimeExpr>>,
        C: Into<Vec<PrimeExpr>>,
        D: Into<Vec<PrimeExpr>>,
        E: Into<Vec<PrimeExpr>>,
        F: Into<PrimeExpr>
    >(var: A, begins: B, axes: C, steps: D, strides: E, val: F) -> Self {
        StoreStmt {
            var: var.into(),
            begins: Arc::new(begins.into()),
            axes: Arc::new(axes.into()),
            steps: Arc::new(steps.into()),
            strides: Arc::new(strides.into()),
            val: Arc::new(val.into()),
        }
    }

    pub fn accept<V: IRVisitor>(&self, visitor: &V) {
        self.var.accept(visitor);
        for begin in self.begins.iter() {
            begin.accept(visitor);
        }
        for step in self.steps.iter() {
            step.accept(visitor);
        }
        for axis in self.axes.iter() {
            axis.accept(visitor);
        }
        for stride in self.strides.iter() {
            stride.accept(visitor);
        }
        self.val.accept(visitor);
    }

    pub fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V) {
        self.var.accept_mut(visitor);
        for begin in self.begins.iter() {
            begin.accept_mut(visitor);
        }
        for step in self.steps.iter() {
            step.accept_mut(visitor);
        }
        for axis in self.axes.iter() {
            axis.accept_mut(visitor);
        }
        for stride in self.strides.iter() {
            stride.accept_mut(visitor);
        }
        self.val.accept_mut(visitor);
    }

    pub fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_store(self);
    }
}

impl std::fmt::Display for StoreStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1i64));
        let zero = PrimeExpr::Int(Int::make(Dtype::I64, 0i64));
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
        write!(
            f,
            "{}{}{}{} {} {};",
            self.var,
            "[".truecolor(220, 105, 173),
            indices,
            "]".truecolor(220, 105, 173),
            "=".purple(),
            self.val
        )
    }
}

impl Into<Stmt> for StoreStmt {
    fn into(self) -> Stmt {
        Stmt::StoreStmt(self)
    }
}

impl Into<Stmt> for &StoreStmt {
    fn into(self) -> Stmt {
        Stmt::StoreStmt(self.clone())
    }
}
