use std::{ ops::{ Deref, DerefMut }, sync::Arc };

use hashbrown::{ HashMap, HashSet };

use crate::{
    halide::{
        exprs::Load,
        ir_cmp::expr_equal,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        substitute::subsititue_var::SubstituteVar,
        traits::{ mutate_expr, AccepterMutate, IRMutVisitor, IRMutateVisitor, MutatorGetSet },
    },
    hlir::tensor_slice::TensorSlice,
    iter_var::IterVar,
};

pub struct SubstituteLoad {
    stmt: Stmt,
    expr: PrimeExpr,
    set: HashSet<Arc<Vec<PrimeExpr>>>,
    to_inline_indices: Arc<Vec<PrimeExpr>>,
    body: PrimeExpr,
    load_var: PrimeExpr,
}

impl SubstituteLoad {
    pub fn new<T: Into<PrimeExpr>>(
        load_var: T,
        to_inline_indices: Arc<Vec<PrimeExpr>>,
        body: PrimeExpr
    ) -> Self {
        SubstituteLoad {
            stmt: Stmt::None,
            expr: PrimeExpr::None,
            set: HashSet::new(),
            load_var: load_var.into(),
            to_inline_indices,
            body,
        }
    }
    pub fn set(&self) -> &HashSet<Arc<Vec<PrimeExpr>>> {
        &self.set
    }
}

impl Deref for SubstituteLoad {
    type Target = HashSet<Arc<Vec<PrimeExpr>>>;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for SubstituteLoad {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
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
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {
        if expr_equal(&slice.name().into(), &self.load_var) {
            let dims = slice.dims_();
            if let Some(dims) = self.set.get(dims) {
                let mut subs_var = SubstituteVar::new();
                for i in dims.iter() {
                    assert!(self.to_inline_indices.contains(i));
                }
                for (a, b) in dims
                    .iter()
                    .zip(self.to_inline_indices.iter().filter(|x| dims.contains(x))) {
                    subs_var.add_replacement(
                        a.to_variable().unwrap().clone(),
                        b.to_variable().unwrap()
                    );
                }
                self.body.accept_mutate(&mut subs_var);
                self.set_expr(subs_var.expr().clone());
                return;
            }
        }
        self.set_expr(slice.clone());
    }
}

pub struct FindInputs {
    vec: Vec<TensorSlice>,
}

impl FindInputs {
    pub fn new() -> Self {
        FindInputs { vec: Vec::new() }
    }
    pub fn to_vec(self) -> Vec<TensorSlice> {
        self.vec
    }
}

impl Deref for FindInputs {
    type Target = Vec<TensorSlice>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl DerefMut for FindInputs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl IRMutVisitor for FindInputs {
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {
        self.vec.push(slice.clone());
    }
}
