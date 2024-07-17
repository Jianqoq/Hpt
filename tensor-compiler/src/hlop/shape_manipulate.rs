use hashbrown::HashSet;
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
            format!("reshape_{}", self.name),
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

    pub fn permute(&self, axes: &[isize]) -> Self {
        let mut new_shape = vec![];
        let mut new_axes = vec![];
        let mut set = HashSet::new();
        for i in axes {
            let mut i = *i;
            if i < 0 {
                i = i + (self.shape.len() as isize);
            } else {
            }
            if let Some(_) = set.get(&i) {
                panic!("Duplicate axes");
            } else {
                set.insert(i);
            }
            if i >= (self.shape.len() as isize) || i < 0 {
                panic!("Invalid axes");
            }
            new_shape.push(self.shape[i as usize].clone());
            new_axes.push(i);
        }
        _compute(
            self.dtype,
            new_shape,
            vec![self.clone()],
            format!("transpose_{}", self.name),
            move |inputs, indices| {
                let mut new_indices = vec![];
                for i in new_axes.iter() {
                    new_indices.push(indices[*i as usize].var().clone());
                }
                inputs[0]._slice(&new_indices).into()
            }
        )
    }

    pub fn squeeze(&self, axes: &[isize]) -> Self {
        let mut new_shape = vec![];
        let mut new_axes = vec![];
        let mut set = HashSet::new();
        for i in axes {
            let mut i = *i;
            if i < 0 {
                i = i + (self.shape.len() as isize);
            } else {
            }
            if let Some(_) = set.get(&i) {
                panic!("Duplicate axes");
            } else {
                set.insert(i);
            }
            if i >= (self.shape.len() as isize) || i < 0 {
                panic!("Invalid axes");
            }
            new_axes.push(i);
        }
        let mut cnt = 0;
        let mut strides = vec![];
        for i in 0..self.shape.len() {
            if let Some(_) = set.get(&(i as isize)) {
                continue;
            }
            let mut iter_var = self.shape[i].clone();
            iter_var.set_var(Variable::new(format!("ax{}", cnt)));
            strides.push(i);
            cnt += 1;
            new_shape.push(iter_var);
        }
        _compute(
            self.dtype,
            new_shape,
            vec![self.clone()],
            format!("reshape_{}", self.name),
            move |inputs, indices| {
                assert!(strides.len() == indices.len());
                let new_indices = indices
                    .iter()
                    .map(|x| x.var().clone())
                    .collect::<Vec<_>>();
                inputs[0]._slice_strides(&new_indices, &strides).into()
            }
        )
    }
}
