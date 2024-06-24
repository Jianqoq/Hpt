use hashbrown::HashMap;

use crate::halide::{
    expr::Expr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
    variable::Variable,
};

pub struct SubstituteVar {
    expr: Expr,
    stmt: Stmt,
    replace: HashMap<Variable, Variable>,
}

impl SubstituteVar {
    pub fn new() -> Self {
        SubstituteVar {
            expr: Expr::None,
            stmt: Stmt::None,
            replace: HashMap::new(),
        }
    }

    fn find_replacement(&self, var: &Variable) -> Option<&Variable> {
        self.replace.get(var)
    }

    pub fn add_replacement<T: Into<Variable>>(&mut self, var: Variable, expr: T) {
        self.replace.insert(var, expr.into());
    }
}

impl IRMutateVisitor for SubstituteVar {
    fn visit_variable(&mut self, var: &Variable) {
        if let Some(replacement) = self.find_replacement(var) {
            self.expr = replacement.clone().into();
        } else {
            self.expr = var.clone().into();
        }
    }
}

impl MutatorGetSet for SubstituteVar {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}
