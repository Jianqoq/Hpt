use std::{ fmt::Display, sync::Arc };

use crate::halide::{ prime_expr::PrimeExpr, variable::Variable };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum IterVar {
    IterVar(_IterVar),
    Splitted(Splitted),
    Fused(Fused),
}

impl IterVar {
    pub fn set_var(&mut self, var: Variable) {
        match self {
            IterVar::IterVar(iter_var) => iter_var.set_var(var),
            _ => panic!("Cannot set var for splitted or fused iter var"),
        }
    }
    pub fn to_iter_var(&self) -> Option<&_IterVar> {
        match self {
            IterVar::IterVar(iter_var) => iter_var.into(),
            _ => None,
        }
    }
    pub fn to_iter_var_mut(&mut self) -> Option<&mut _IterVar> {
        match self {
            IterVar::IterVar(iter_var) => iter_var.into(),
            _ => None,
        }
    }
    pub fn to_splitted(&self) -> Option<&Splitted> {
        match self {
            IterVar::Splitted(splitted) => splitted.into(),
            _ => None,
        }
    }
    pub fn to_splitted_mut(&mut self) -> Option<&mut Splitted> {
        match self {
            IterVar::Splitted(splitted) => splitted.into(),
            _ => None,
        }
    }
    pub fn to_fused(&self) -> Option<&Fused> {
        match self {
            IterVar::Fused(fused) => fused.into(),
            _ => None,
        }
    }
    pub fn to_fused_mut(&mut self) -> Option<&mut Fused> {
        match self {
            IterVar::Fused(fused) => fused.into(),
            _ => None,
        }
    }

    pub fn to_prime_expr(&self) -> PrimeExpr {
        match self {
            IterVar::IterVar(iter_var) => iter_var.var().clone().into(),
            IterVar::Splitted(splitted) => splitted.to_prime_expr(),
            IterVar::Fused(fused) => fused.to_prime_expr(),
        }
    }

    pub fn real_axes(&self) -> Vec<IterVar> {
        match self {
            IterVar::IterVar(_) => vec![self.clone()],
            IterVar::Splitted(splitted) => {
                let mut outer = splitted.outer.real_axes();
                let factor = splitted.inner.real_axes();
                outer.extend(factor);
                outer
            }
            IterVar::Fused(_) => vec![self.clone()],
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Splitted {
    pub(crate) outer: Arc<IterVar>,
    pub(crate) inner: Arc<IterVar>,
    pub(crate) factor: PrimeExpr,
    pub(crate) var: Variable,
}

impl Splitted {
    pub fn new<A: Into<IterVar>, B: Into<IterVar>, C: Into<PrimeExpr>>(
        outer: A,
        inner: B,
        factor: C,
        var: Variable
    ) -> Self {
        Self {
            outer: Arc::new(outer.into()),
            inner: Arc::new(inner.into()),
            factor: factor.into(),
            var,
        }
    }
    pub fn to_prime_expr(&self) -> PrimeExpr {
        match self.outer.as_ref() {
            IterVar::IterVar(var) => {
                let var: PrimeExpr = var.var.clone().into();
                (var + (&self.factor - 1)).floor_div(&self.factor)
            }
            IterVar::Splitted(splitted) => {
                let outer = splitted.outer.to_prime_expr();
                (outer + (&self.factor - 1)).floor_div(&self.factor)
            }
            IterVar::Fused(fused) => {
                let outer = fused.to_prime_expr();
                (outer + (&self.factor - 1)).floor_div(&self.factor)
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Fused {
    pub(crate) axis1: Arc<IterVar>,
    pub(crate) axis2: Arc<IterVar>,
    pub(crate) var: Variable,
}

impl Fused {
    pub fn new<A: Into<IterVar>, B: Into<IterVar>, C: Into<Variable>>(
        axis1: A,
        axis2: B,
        var: C
    ) -> Self {
        Self {
            axis1: Arc::new(axis1.into()),
            axis2: Arc::new(axis2.into()),
            var: var.into(),
        }
    }
    pub fn to_prime_expr(&self) -> PrimeExpr {
        todo!()
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct _IterVar {
    start: PrimeExpr,
    end: PrimeExpr,
    step: PrimeExpr,
    var: Variable,
}

impl _IterVar {
    pub fn new<A: Into<PrimeExpr>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<Variable>>(
        start: A,
        end: B,
        step: C,
        var: D
    ) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
            step: step.into(),
            var: var.into(),
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

impl<A, B, C, D> Into<IterVar>
    for (A, B, C, D)
    where A: Into<PrimeExpr>, B: Into<PrimeExpr>, C: Into<PrimeExpr>, D: Into<Variable>
{
    fn into(self) -> IterVar {
        IterVar::IterVar(_IterVar::new(self.0, self.1, self.2, self.3.into()))
    }
}

impl Into<_IterVar> for &_IterVar {
    fn into(self) -> _IterVar {
        self.clone()
    }
}

impl Into<IterVar> for _IterVar {
    fn into(self) -> IterVar {
        IterVar::IterVar(self)
    }
}

impl Into<IterVar> for &_IterVar {
    fn into(self) -> IterVar {
        IterVar::IterVar(self.clone())
    }
}

impl Into<IterVar> for &IterVar {
    fn into(self) -> IterVar {
        self.clone()
    }
}

impl Into<IterVar> for Fused {
    fn into(self) -> IterVar {
        IterVar::Fused(self)
    }
}

impl Into<IterVar> for &Fused {
    fn into(self) -> IterVar {
        IterVar::Fused(self.clone())
    }
}

impl Display for _IterVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "for {} in ({}..{}).step_by({})", self.var, self.start, self.end, self.step)
    }
}
