use std::ops::{ Deref, DerefMut };

use hashbrown::HashMap;

use crate::halide::{
    exprs::Load,
    ir_cmp::expr_equal,
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ mutate_expr, IRMutateVisitor, MutatorGetSet },
};

pub struct SubstituteLoad {
    stmt: Stmt,
    expr: PrimeExpr,
    map: HashMap<PrimeExpr, PrimeExpr>,
    load_var: PrimeExpr,
}

impl SubstituteLoad {
    pub fn new<T: Into<PrimeExpr>>(load_var: T) -> Self {
        SubstituteLoad {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            map: HashMap::new(),
            load_var: load_var.into(),
        }
    }
}

impl Deref for SubstituteLoad {
    type Target = HashMap<PrimeExpr, PrimeExpr>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for SubstituteLoad {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

impl MutatorGetSet for SubstituteLoad {
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

impl IRMutateVisitor for SubstituteLoad {
    fn mutate_expr(&mut self, expr: &PrimeExpr) -> PrimeExpr {
        if let Some(replace) = self.map.get(expr) {
            return replace.clone();
        } else {
            return mutate_expr(self, expr);
        }
    }

    fn visit_load(&mut self, load: &Load) {
        if expr_equal(load.name(), &self.load_var) {
            let indices = self.mutate_expr(load.indices());
            self.set_expr(Load::make(load.name(), &indices));
        } else {
            self.set_expr(load);
        }
    }
}
