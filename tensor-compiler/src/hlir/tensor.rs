#![allow(unused_imports)]
use std::sync::Arc;

use hashbrown::HashMap;
use tensor_common::shape::Shape;
use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::{ Int, Load },
    let_stmt::LetStmt,
    loop_utils::build_nested::build_nested_for,
    prime_expr::PrimeExpr,
    seq_stmt::Seq,
    stmt::Stmt,
    store_stmt::StoreStmt,
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
            let name = tensor.name.clone();
            let expr = op(
                loop_indexes
                    .iter()
                    .map(|x| x.clone().into())
                    .collect::<Vec<_>>()
            );
            let mut main_stmt: Vec<Stmt> = vec![];
            match expr {
                PrimeExpr::Reduce(reduce) => {
                    let indices = &loop_indexes
                        .iter()
                        .map(|x| x.clone().into())
                        .reduce(|acc, x| acc + x)
                        .unwrap();
                    let loop_vars = reduce
                        .loop_var()
                        .iter()
                        .map(|x| x.to_variable().unwrap().clone())
                        .collect::<Vec<_>>();
                    let end = reduce.end();
                    let fors = match reduce.op() {
                        "sum" => {
                            assert!(reduce.identity().len() == 1);
                            assert!(reduce.expr().len() == 1);
                            main_stmt.push(
                                StoreStmt::make(
                                    &Variable::make(&format!("output_{}", name)),
                                    indices,
                                    &reduce.identity()[0].clone()
                                ).into()
                            );
                            build_nested_for(
                                &loop_vars,
                                end,
                                StoreStmt::make(
                                    &Variable::make(&format!("output_{}", name)),
                                    &loop_indexes
                                        .iter()
                                        .map(|x| x.clone().into())
                                        .reduce(|acc, x| acc + x)
                                        .unwrap(),
                                    &reduce.expr()[0]
                                )
                            )
                        }
                        "min" => {
                            todo!();
                        }
                        "max" => {
                            todo!();
                        }
                        "prod" => {
                            todo!();
                        }
                        "argmin" => {
                            todo!();
                        }
                        "argmax" => {
                            todo!();
                        }
                        _ => todo!(),
                    };
                    main_stmt.push(fors);
                }
                _ => todo!(),
            }
            let loop_stmt = build_nested_for(
                &loop_indexes,
                &shape,
                Stmt::Seq(Seq::make(main_stmt))
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
    use crate::halide::{
        exprs::Int,
        loop_utils::sum::sum,
        prime_expr::PrimeExpr,
        printer::IRPrinter,
    };

    #[test]
    fn test_reduce() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let a = Tensor::placeholder(vec![n.clone().into(), m.clone().into()], "a");
        let a_op = a.op.clone();
        let c = compute(vec![n.clone().into()], "c", move |vec| {
            sum(
                vec![a_op(vec![vec[0].clone(), Variable::make("k").into()])],
                vec![Int::make(Dtype::BF16, 0).into()],
                vec![Int::make(Dtype::BF16, 0).into()],
                vec![m.clone().into()],
                vec![Int::make(Dtype::BF16, 1).into()],
                vec!["k"]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
}
