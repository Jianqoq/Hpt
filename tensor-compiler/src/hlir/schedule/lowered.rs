use std::{ ops::{ Deref, DerefMut }, sync::Arc };

use hashbrown::{ HashMap, HashSet };

use crate::{
    halide::{
        exprs::{ FloorDiv, Load, Mul },
        ir_cmp::expr_equal,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        substitute::{ subsititue_expr::SubstituteExpr, subsititue_var::SubstituteVar },
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
                assert!(target_dims.len() == self.to_inline_indices.len());
                let mut subs_expr = SubstituteExpr::new();
                for (idx, (inline_dim, target_dim)) in self.to_inline_indices
                    .iter()
                    .zip(target_dims.iter())
                    .enumerate() {
                    // indice could be splitted, fused, normal
                    // however, the len of to_inline_indices is always equal to the len of target_dims
                    match inline_dim {
                        IterVar::IterVar(iter_var) => {
                            subs_expr.add_replacement(iter_var.var().to_prime_expr(), target_dim.clone());
                        }
                        IterVar::Splitted(splitted) => {
                            let outer = (target_dim + (&splitted.factor - 1)).floor_div(
                                &splitted.factor
                            );
                            subs_expr.add_replacement(splitted.to_prime_expr(), outer);
                        }
                        IterVar::Fused(fused) => {
                            let iter_var2 = fused.axis2.to_prime_expr();
                            let var = fused.var.to_prime_expr();
                            let i = PrimeExpr::FloorDiv(FloorDiv::make(&var, &iter_var2));
                            let j = &var % &iter_var2;

                            let prev_inline_indices = &self.to_inline_indices[idx - 1];
                            if prev_inline_indices == inline_dim {
                                let prev_target_dim = &target_dims[idx - 1];
                                let mul = prev_target_dim * target_dim;
                                let new_i = PrimeExpr::FloorDiv(FloorDiv::make(&mul, &target_dim));
                                let new_j = &mul % &target_dim;
                                subs_expr.add_replacement(i, new_i);
                                subs_expr.add_replacement(j, new_j);
                            } else {
                                assert!(self.to_inline_indices.len() > idx + 1);
                                assert_eq!(&self.to_inline_indices[idx + 1], inline_dim);
                                let next_target_dim = &target_dims[idx + 1];
                                let mul = target_dim * next_target_dim;
                                let new_i = PrimeExpr::FloorDiv(
                                    FloorDiv::make(&mul, &next_target_dim)
                                );
                                let new_j = &mul % &next_target_dim;
                                subs_expr.add_replacement(i, new_i);
                                subs_expr.add_replacement(j, new_j);
                            }
                        }
                    }
                }
                self.body.accept_mutate(&mut subs_expr);
                self.set_expr(subs_expr.expr().clone());
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
