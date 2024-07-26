use std::sync::Arc;

use crate::halide::{
    exprs::Load,
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
    variable::Variable,
};

pub struct InsertAxes {
    pub(crate) stmt: Stmt,
    pub(crate) expr: PrimeExpr,
    pub(crate) axes: Arc<Vec<PrimeExpr>>,
    pub(crate) id: usize,
}

impl InsertAxes {
    pub fn new(axes: Arc<Vec<PrimeExpr>>, id: usize) -> Self {
        Self {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            axes,
            id,
        }
    }
}

impl MutatorGetSet for InsertAxes {
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

impl IRMutateVisitor for InsertAxes {
    fn visit_tensor_load(&mut self, tensor_load: &crate::halide::tensor_load::TensorLoad) {
        let mut new_begins = (0..self.axes.len())
            .map(|_| (0i64).into())
            .collect::<Vec<PrimeExpr>>();
        let mut new_steps = (0..self.axes.len()).map(|_| (1i64).into()).collect::<Vec<PrimeExpr>>();
        let mut new_strides = (0..self.axes.len())
            .map(|idx|
                PrimeExpr::Load(Load::make(Variable::new(format!("%{}.s", self.id)), idx as i64))
            )
            .collect::<Vec<PrimeExpr>>();
        for (idx, stride) in tensor_load.strides.iter().enumerate() {
            let mut load = stride.to_load().unwrap().clone();
            load.indices = Arc::new(((idx as i64) + (self.axes.len() as i64)).into());
            new_strides.push(load.into());
        }
        for (idx, begin) in tensor_load.begins.iter().enumerate() {
            let mut load = begin.to_load().unwrap().clone();
            load.indices = Arc::new(((idx as i64) + (self.axes.len() as i64)).into());
            new_begins.push(load.into());
        }
        for (idx, step) in tensor_load.steps.iter().enumerate() {
            let mut load = step.to_load().unwrap().clone();
            load.indices = Arc::new(((idx as i64) + (self.axes.len() as i64)).into());
            new_steps.push(load.into());
        }
        self.set_expr(
            crate::halide::tensor_load::TensorLoad::make(
                tensor_load.var.as_ref(),
                new_begins,
                self.axes.as_ref().clone(),
                new_steps,
                new_strides,
                tensor_load.hints.as_ref().clone()
            )
        );
    }
}
