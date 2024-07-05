use std::{ ops::{ Deref, DerefMut }, sync::Arc };

use hashbrown::{ HashMap, HashSet };

use crate::{
    halide::{
        exprs::{ Load, Mul },
        ir_cmp::expr_equal,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        substitute::subsititue_var::SubstituteVar,
        traits::{ mutate_expr, AccepterMutate, IRMutVisitor, IRMutateVisitor, MutatorGetSet },
    },
    hlir::tensor_slice::TensorSlice,
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

pub struct SubstituteLoad {
    stmt: Stmt,
    expr: PrimeExpr,
    set: HashSet<Arc<Vec<PrimeExpr>>>,
    to_inline_indices: Arc<Vec<IterVar>>,
    body: PrimeExpr,
    load_var: PrimeExpr,
}

impl SubstituteLoad {
    pub fn new<T: Into<PrimeExpr>>(
        load_var: T,
        to_inline_indices: Arc<Vec<IterVar>>,
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
            if let Some(target_dims) = self.set.get(dims) {
                assert!(target_dims.len() == self.to_inline_indices.len());
                let mut subs_var = SubstituteVar::new();
                assert!(target_dims.len() == self.to_inline_indices.len());
                let mut map = HashMap::new();
                for (inline_dim, target_dim) in self.to_inline_indices
                    .iter()
                    .zip(target_dims.iter()) {
                    // indice could be splitted, fused, normal
                    // however, the len of to_inline_indices is always equal to the len of target_dims
                    match inline_dim {
                        IterVar::IterVar(iter_var) => {
                            map.insert(iter_var.var().to_prime_expr(), target_dim.clone());
                        }
                        IterVar::Splitted(splitted) => {
                            let outter = &splitted.outer;
                            let inner = &splitted.inner;
                            let outer = (target_dim + (&splitted.factor - 1)).floor_div(
                                &splitted.factor
                            );
                        }
                        IterVar::Fused(_) => todo!(),
                    }
                }
                // self.body.accept_mutate(&mut subs_var);
                // self.set_expr(subs_var.expr().clone());
                // return;
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
