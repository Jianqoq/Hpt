use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ FloorDiv, Int }, prime_expr::PrimeExpr, variable::Variable },
    hlir::tensor::{ Tensor, _compute },
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

pub fn flatten_index(new_indices: &[Variable], new_shape: &[PrimeExpr]) -> PrimeExpr {
    if new_indices.len() == 0 {
        return PrimeExpr::Int(Int::make(Dtype::I64, 0));
    }
    let mut idx = PrimeExpr::None;
    let rev_shape = new_shape.iter().rev().collect::<Vec<_>>();
    for i in 0..new_shape.len() {
        if i == 0 {
            idx = new_indices[i].clone().into();
        } else {
            let indice: PrimeExpr = new_indices[i].clone().into();
            idx = idx * rev_shape[i].clone() + indice;
        }
    }
    idx
}
fn unflatten_index(old_shape: &[PrimeExpr], flat_idx: &PrimeExpr) -> Vec<PrimeExpr> {
    let mut indices = vec![];
    let mut flat_idx = flat_idx.clone();
    for dim in old_shape.iter().rev() {
        indices.push(&flat_idx % dim);
        flat_idx = PrimeExpr::FloorDiv(FloorDiv::make(&flat_idx, dim));
    }
    indices.reverse();
    indices
}

impl Tensor {
    pub fn reshape(&self, shape: &[&dyn ToPrimeExpr]) -> Self {
        let res_shape = shape
            .iter()
            .enumerate()
            .map(|(idx, x)|
                IterVar::new(0, x.to_prime_expr(), 1i64, Variable::new(format!("ax{}", idx)))
            )
            .collect::<Vec<_>>();
        let old_shape = self.shape
            .iter()
            .map(|x| x.end().into())
            .collect::<Vec<_>>();
        let new_shape = shape
            .iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<_>>();
        _compute(
            self.dtype,
            res_shape,
            vec![self.clone()],
            &format!("reshape_{}", self.name),
            move |inputs, indices| {
                let flat_idx = flatten_index(
                    &indices
                        .iter()
                        .map(|x| x.var().clone())
                        .collect::<Vec<_>>(),
                    &new_shape
                );
                let indices = unflatten_index(&old_shape, &flat_idx);
                inputs[0]._slice(&indices).into()
            }
        )
    }
}
