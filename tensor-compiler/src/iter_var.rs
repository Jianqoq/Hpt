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
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Splitted {
    pub(crate) outer: Arc<IterVar>,
    pub(crate) inner: Arc<IterVar>,
    pub(crate) correspond: PrimeExpr,
}

impl Splitted {
    pub fn new<A: Into<IterVar>, B: Into<IterVar>, C: Into<PrimeExpr>>(
        outer: A,
        inner: B,
        correspond: C
    ) -> Self {
        Self {
            outer: Arc::new(outer.into()),
            inner: Arc::new(inner.into()),
            correspond: correspond.into(),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Fused {
    pub(crate) iter_var: Arc<IterVar>,
    pub(crate) corresponds: Arc<[PrimeExpr; 2]>,
}

impl Fused {
    pub fn new<A: Into<IterVar>, B: Into<[PrimeExpr; 2]>>(
        iter_var: A,
        corresponds: B
    ) -> Self {
        Self {
            iter_var: Arc::new(iter_var.into()),
            corresponds: Arc::new(corresponds.into()),
        }
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
