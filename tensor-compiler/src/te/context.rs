use std::{ collections::HashMap, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::Int, prime_expr::PrimeExpr, variable::Variable },
    hlir::tensor_slice::TensorSlice,
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

use super::{ rc_mut::RcMut, tensor::Tensor };

#[derive(Clone)]
pub struct Context {
    nodes: RcMut<HashMap<usize, Tensor>>,
    id: RcMut<usize>,
}

impl Context {
    pub fn placeholder(&mut self, shape: &[&dyn ToPrimeExpr]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let iter_vars = shape
            .into_iter()
            .map(|x| x.to_prime_expr())
            .enumerate()
            .map(|(i, x)| {
                IterVar::new(
                    Int::make(Dtype::I64, 0i64),
                    x,
                    Int::make(Dtype::I64, 1i64),
                    Variable::new(format!("ax{}", i))
                )
            })
            .collect::<Vec<_>>();
        let tensor = Tensor {
            shape: shape
                .into_iter()
                .map(|x| x.to_prime_expr())
                .collect(),
            ctx: self.clone(),
            body: TensorSlice::make(
                Variable::new(format!("%{}", id)),
                iter_vars
                    .iter()
                    .map(|x| x.var().to_prime_expr())
                    .collect::<Vec<PrimeExpr>>()
            ).into(),
            inputs: Arc::new(vec![]),
            id,
        };
        self.nodes.borrow_mut().insert(id, tensor.clone());
        tensor
    }
}
