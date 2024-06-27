#![allow(dead_code)]
use hashbrown::HashMap;

use crate::{
    halide::variable::Variable,
    hlir::{ expr::Expr, exprs::ComputeNode, traits::{ HlirMutateVisitor, MutatorGetSet } },
};

pub struct FuseComputeNode {
    map: HashMap<Variable, ComputeNode>,
    expr: Expr,
}

impl MutatorGetSet for FuseComputeNode {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }
}

impl HlirMutateVisitor for FuseComputeNode {
    fn visit_let(&mut self, let_: &crate::hlir::exprs::Let) {
        let var = let_.var();
        let value = let_.value();
        let body = let_.body();

        self.visit_expr(body);
    }
}
