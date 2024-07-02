use crate::halide::{ prime_expr::PrimeExpr, variable::Variable };

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct IterVar {
    start: PrimeExpr,
    end: PrimeExpr,
    step: PrimeExpr,
    var: Variable,
}

impl IterVar {
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
        IterVar::new(self.0, self.1, self.2, self.3.into())
    }
}
