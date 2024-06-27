use hashbrown::HashMap;

use crate::{
    halide::{
        prime_expr::PrimeExpr,
        exprs::Int,
        for_stmt::For,
        stmt::Stmt,
        traits::{ IRMutateVisitor, MutatorGetSet },
        variable::Variable,
    },
    I64_TYPE,
};

pub struct SubstituteForMeta {
    replace: HashMap<(i64, i64, Variable), (i64, i64, Variable)>,
    stmt: Stmt,
    expr: PrimeExpr,
}

impl SubstituteForMeta {
    pub fn new() -> Self {
        SubstituteForMeta {
            replace: HashMap::new(),
            stmt: Stmt::None,
            expr: PrimeExpr::None,
        }
    }

    fn find_replacement(&self, key: &(i64, i64, Variable)) -> Option<&(i64, i64, Variable)> {
        self.replace.get(key)
    }

    pub fn add_replacement<T: Into<Variable>>(
        &mut self,
        old_start: i64,
        old_end: i64,
        old_var: Variable,
        new_start: i64,
        new_end: i64,
        new_var: T
    ) {
        self.replace.insert((old_start, old_end, old_var), (new_start, new_end, new_var.into()));
    }
}

impl IRMutateVisitor for SubstituteForMeta {
    fn visit_for(&mut self, for_stmt: &For) {
        if
            let Some((new_start, new_end, new_var)) = self.find_replacement(
                &(
                    for_stmt.start().to_int().unwrap().value(),
                    for_stmt.end().to_int().unwrap().value(),
                    for_stmt.var().clone(),
                )
            )
        {
            self.stmt = For::make(
                new_var,
                Int::make(I64_TYPE, *new_start),
                Int::make(I64_TYPE, *new_end),
                for_stmt.stmt()
            ).into();
        } else {
            self.stmt = for_stmt.clone().into();
        }
    }
}

impl MutatorGetSet for SubstituteForMeta {
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
