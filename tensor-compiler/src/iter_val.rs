use std::{ fmt::Display, sync::Arc };

use crate::halide::{ prime_expr::PrimeExpr, variable::Variable };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum IterVar {
    IterVar(_IterVar),
    Splitted(Splitted),
    Fused(Fused),
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Splitted {
    pub(crate) outer: Arc<IterVar>,
    pub(crate) inner: Arc<IterVar>,
    pub(crate) correspond: PrimeExpr,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Fused {
    pub(crate) iter_var: Arc<IterVar>,
    pub(crate) corresponds: Arc<[PrimeExpr; 2]>,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct _IterVar {
    start: PrimeExpr,
    end: PrimeExpr,
    step: PrimeExpr,
    var: Variable,
}

impl _IterVar {
    pub fn new<A: Into<PrimeExpr>, B: Into<PrimeExpr>, C: Into<PrimeExpr>>(
        start: A,
        end: B,
        step: C,
        var: Variable
    ) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            step: step.into(),
            var,
        }
    }

    pub fn start(&self) -> &PrimeExpr {
        &self.start
    }

    pub fn end(&self) -> &PrimeExpr {
        &self.end
    }

    pub fn step(&self) -> &PrimeExpr {
        &self.step
    }

    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn set_var(&mut self, var: Variable) {
        self.var = var;
    }
}

impl<A, B, C, D> Into<_IterVar>
    for (A, B, C, D)
    where A: Into<PrimeExpr>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<Variable>
{
    fn into(self) -> _IterVar {
        _IterVar::new(self.0, self.1, self.2, self.3.into())
    }
}

impl Into<_IterVar> for &_IterVar {
    fn into(self) -> _IterVar {
        self.clone()
    }
}

impl Display for _IterVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "for {} in ({}..{}).step_by({})", self.var, self.start, self.end, self.step)
    }
}
