use std::sync::Arc;

use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::{ FloorDiv, Int },
        prime_expr::PrimeExpr,
        stmt::Stmt,
        traits::{ AccepterMut, AccepterMutate, IRMutateVisitor, MutatorGetSet },
        variable::Variable,
    },
    hlir::{ schedule::lowered::FindInputs, tensor::Tensor, tensor_slice::TensorSlice },
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

pub struct ReshapeVisitor {
    pub stmt: Stmt,
    pub expr: PrimeExpr,
    pub new_shape: Vec<PrimeExpr>,
    pub old_shape: Vec<PrimeExpr>,
    pub new_indices: Vec<Variable>,
}

impl ReshapeVisitor {
    pub fn new() -> Self {
        Self {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            new_shape: vec![],
            old_shape: vec![],
            new_indices: vec![],
        }
    }

    pub fn set_new_shape(&mut self, new_shape: Vec<PrimeExpr>) {
        self.new_shape = new_shape;
    }
    pub fn set_old_shape(&mut self, old_shape: Vec<PrimeExpr>) {
        self.old_shape = old_shape;
    }
    pub fn set_new_indices(&mut self, new_indices: Vec<Variable>) {
        self.new_indices = new_indices;
    }

    pub fn flatten_index(&self) -> PrimeExpr {
        if self.new_indices.len() == 0 {
            return PrimeExpr::Int(Int::make(Dtype::I64, 0));
        }
        let mut idx = PrimeExpr::None;
        let rev_shape = self.new_shape.iter().rev().collect::<Vec<_>>();
        for i in 0..self.new_shape.len() {
            if i == 0 {
                idx = self.new_indices[i].clone().into();
            } else {
                let indice: PrimeExpr = self.new_indices[i].clone().into();
                idx = idx * rev_shape[i].clone() + indice;
            }
        }
        idx
    }
    fn unflatten_index(&self, flat_idx: &PrimeExpr) -> Vec<PrimeExpr> {
        let mut indices = vec![];
        let mut flat_idx = flat_idx.clone();
        for dim in self.old_shape.iter().rev() {
            indices.push(&flat_idx % dim);
            flat_idx = PrimeExpr::FloorDiv(FloorDiv::make(&flat_idx, dim));
        }
        indices.reverse();
        indices
    }
}

impl MutatorGetSet for ReshapeVisitor {
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

impl IRMutateVisitor for ReshapeVisitor {
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {
        let flatten_idx = self.flatten_index();
        self.set_expr(TensorSlice::make(slice.name().clone(), self.unflatten_index(&flatten_idx)));
    }
}

impl Tensor {
    pub fn reshape(&self, shape: &[&dyn ToPrimeExpr]) -> Self {
        let shape = shape
            .iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<_>>();
        let mut visitor = ReshapeVisitor::new();
        visitor.set_new_shape(shape.clone());
        visitor.set_old_shape(
            self.shape
                .iter()
                .map(|x| x.end().clone())
                .collect()
        );
        let shape = shape
            .iter()
            .enumerate()
            .map(|(idx, x)| IterVar::new(0, x, 1i64, Variable::new(format!("ax{}", idx))))
            .collect::<Vec<_>>();
        visitor.set_new_indices(
            shape
                .iter()
                .map(|x| x.var().clone())
                .collect()
        );
        self.body.accept_mutate(&mut visitor);
        let mut input_visitor = FindInputs::new();
        visitor.expr.accept_mut(&mut input_visitor);
        Tensor {
            shape: shape.into(),
            body: visitor.expr.clone(),
            name: Arc::new(format!("reshape_{}", self.name)),
            inputs: input_visitor.to_vec().into(),
            dtype: self.dtype,
        }
    }
}
