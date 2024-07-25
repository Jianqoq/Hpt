use crate::halide::{ prime_expr::PrimeExpr, stmt::Stmt, traits::MutatorGetSet };

pub struct StridesVisitor {
    pub(crate) cnt: i64,
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
}

impl StridesVisitor {
    pub fn new() -> Self {
        Self {
            cnt: 0,
            expr: PrimeExpr::None,
            stmt: Stmt::None,
        }
    }
}

impl MutatorGetSet for StridesVisitor {
    fn set_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &PrimeExpr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}
