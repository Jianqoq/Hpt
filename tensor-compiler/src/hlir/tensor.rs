#![allow(unused_imports)]
use std::sync::Arc;

use hashbrown::HashMap;
use tensor_common::shape::Shape;

use crate::halide::{
    exprs::Load,
    let_stmt::LetStmt,
    loop_utils::build_nested::build_nested_for,
    prime_expr::PrimeExpr,
    seq_stmt::Seq,
    stmt::Stmt,
    variable::Variable,
};

#[derive(Clone)]
pub struct Tensor {
    shape: Arc<Vec<PrimeExpr>>,
    op: Arc<dyn Fn(Vec<PrimeExpr>) -> PrimeExpr>,
    name: Arc<String>,
}

impl Eq for Tensor {}

impl std::hash::Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        let ptr = Arc::as_ptr(&self.op);
        ptr.hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && Arc::ptr_eq(&self.op, &other.op) && self.name == other.name
    }
}

impl Tensor {
    pub fn placeholder(shape: Vec<PrimeExpr>, name: &str) -> Self {
        let tensor_name = name.to_string();
        Self {
            shape: Arc::new(shape),
            op: Arc::new(move |vec| {
                Load::make(
                    Variable::new(tensor_name.to_string()),
                    vec
                        .iter()
                        .map(|x| x.clone())
                        .reduce(|acc, x| acc + x)
                        .unwrap()
                ).into()
            }),
            name: name.to_string().into(),
        }
    }
}

pub fn compute<F>(res_shape: Vec<PrimeExpr>, name: &str, op: F) -> Tensor
    where F: Fn(Vec<PrimeExpr>) -> PrimeExpr + 'static
{
    Tensor {
        shape: Arc::new(res_shape),
        op: Arc::new(op),
        name: name.to_string().into(),
    }
}

pub struct Schedule {
    ops: HashMap<Tensor, Arc<dyn Fn(Vec<PrimeExpr>) -> PrimeExpr>>,
}

impl Schedule {
    pub fn create(tensors: Vec<Tensor>) -> Self {
        let mut ops = HashMap::new();
        for tensor in tensors {
            let op = tensor.op.clone();
            ops.insert(tensor, op);
        }
        Self { ops }
    }
    pub fn lower(&self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        for (tensor, op) in &self.ops {
            let shape = tensor.shape.clone();
            let loop_indexes = (0..shape.len())
                .map(|i| Variable::new(format!("i{}", i)))
                .collect::<Vec<_>>();
            let expr = op(
                loop_indexes
                    .iter()
                    .map(|x| x.clone().into())
                    .collect::<Vec<_>>()
            );
            let loop_stmt = build_nested_for(
                &loop_indexes,
                &shape,
                Stmt::Seq(Seq::make(vec![LetStmt::make(&Variable::make("test"), expr).into()]))
            );
            stmts.push(loop_stmt);
        }
        stmts
    }
}

#[cfg(test)]
mod tests {
    use tensor_types::dtype::Dtype;

    use super::*;
    use crate::halide::{ exprs::Int, prime_expr::PrimeExpr, printer::IRPrinter };

    #[test]
    fn test_tensor() {
        let a = Tensor::placeholder(
            vec![
                Int::make(Dtype::I64, 50).into(),
                Int::make(Dtype::I64, 3).into(),
                Int::make(Dtype::I64, 8).into(),
                Int::make(Dtype::I64, 8).into()
            ],
            "a"
        );
        let b = Tensor::placeholder(
            vec![
                Int::make(Dtype::I64, 50).into(),
                Int::make(Dtype::I64, 3).into(),
                Int::make(Dtype::I64, 8).into(),
                Int::make(Dtype::I64, 8).into()
            ],
            "b"
        );
        let a_op = a.op.clone();
        let b_op = b.op.clone();
        let c = compute(
            vec![
                Int::make(Dtype::I64, 50).into(),
                Int::make(Dtype::I64, 3).into(),
                Int::make(Dtype::I64, 8).into(),
                Int::make(Dtype::I64, 8).into()
            ],
            "c",
            move |vec| { a_op(vec.clone()) + b_op(vec) }
        );
        let schedule = Schedule::create(vec![a, b, c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
}
