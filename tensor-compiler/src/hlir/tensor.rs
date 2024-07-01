#![allow(unused_imports)]
use std::sync::Arc;

use hashbrown::HashMap;
use tensor_common::shape::Shape;
use tensor_types::dtype::Dtype;

use crate::{
    halide::{
        exprs::{ Add, Int, Load },
        let_stmt::LetStmt,
        loop_utils::build_nested::build_nested_for,
        prime_expr::PrimeExpr,
        seq_stmt::Seq,
        stmt::Stmt,
        store_stmt::StoreStmt,
        variable::Variable,
    },
    iter_val::IterVar,
};

#[derive(Clone)]
pub struct Tensor {
    shape: Arc<Vec<IterVar>>,
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
    pub fn placeholder<T: IntoIterator<Item: Into<PrimeExpr>>>(shape: T, name: &str) -> Self {
        let tensor_name = name.to_string();
        let iter_vars = shape
            .into_iter()
            .map(|x| x.into())
            .enumerate()
            .map(|(i, x)| {
                IterVar::new(
                    Int::make(Dtype::I64, 0),
                    x,
                    Int::make(Dtype::I64, 1),
                    Variable::new(format!("i{}", i))
                )
            })
            .collect::<Vec<_>>();
        Self {
            shape: Arc::new(iter_vars),
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
    pub fn slice<T: IntoIterator<Item: Into<PrimeExpr>>>(&self, indices: T) -> PrimeExpr {
        Load::make(
            Variable::make(&self.name),
            indices
                .into_iter()
                .map(|x| x.into())
                .reduce(|acc, x| acc + x)
                .unwrap()
        ).into()
    }
}

pub fn compute<const N: usize, F>(res_shape: [PrimeExpr; N], name: &str, op: F) -> Tensor
    where F: Fn([PrimeExpr; N]) -> PrimeExpr + 'static
{
    let new_fn = move |vec: Vec<PrimeExpr>| -> PrimeExpr { op(vec.try_into().unwrap()) };
    let iter_vars = res_shape
        .iter()
        .enumerate()
        .map(|(i, x)| {
            IterVar::new(
                Int::make(Dtype::I64, 0),
                x.clone(),
                Int::make(Dtype::I64, 1),
                Variable::new(format!("i{}", i))
            )
        })
        .collect::<Vec<_>>();
    Tensor {
        shape: Arc::new(iter_vars),
        op: Arc::new(new_fn),
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
            let name = tensor.name.clone();
            let expr = op(
                shape
                    .iter()
                    .map(|x| x.var().clone().into())
                    .collect()
            );
            let mut main_stmt: Vec<Stmt> = vec![];
            match expr {
                PrimeExpr::Reduce(reduce) => {
                    let indices = &shape
                        .iter()
                        .map(|x| x.var().into())
                        .reduce(|acc: PrimeExpr, x| acc + x)
                        .unwrap();
                    let iter_vars = reduce.iter_vars();
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
                                iter_vars,
                                StoreStmt::make(
                                    &Variable::make(&format!("output_{}", name)),
                                    indices,
                                    &Add::make(
                                        &Load::make(
                                            Variable::make(&format!("output_{}", name)),
                                            indices
                                        ),
                                        &reduce.expr()[0]
                                    )
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
            let loop_stmt = build_nested_for(&shape, Stmt::Seq(Seq::make(main_stmt)));
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
        let a = Tensor::placeholder([&n, &m], "a");
        let a_op = a.op.clone();
        let c = compute([n.clone().into()], "c", move |[i]| {
            sum(
                [a_op(vec![i, Variable::make("k").into()])],
                [Int::make(Dtype::BF16, 0)],
                [(0, &m, 1, "k")]
            )
        });
        let schedule = Schedule::create(vec![c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }

    #[test]
    fn test_nested_reduce() {
        let n = Variable::make("n");
        let m = Variable::make("m");
        let u = Variable::make("u");
        let end1 = Variable::make("end1");
        let end2 = Variable::make("end2");
        let end3 = Variable::make("end3");
        let a = Tensor::placeholder([&u, &n, &m], "a");
        let _a = a.clone();
        let c = compute([u.clone().into(), n.clone().into()], "c", move |[u, n]| {
            sum(
                [_a.slice([u, n, Variable::make("k").into()])],
                [0],
                [
                    (0, &end1, 1, "k"),
                    (0, &end2, 1, "n"),
                ]
            )
        });
        let _c = c.clone();
        let d = compute([u.clone().into()], "d", move |[i]| {
            sum(
                [_c.slice([i, Variable::make("j").into()])],
                [Int::make(Dtype::BF16, 0)],
                [(0, &end3, 1, "j")]
            )
        });
        let schedule = Schedule::create(vec![d, c]);
        let lowered = schedule.lower();
        for stmt in lowered {
            IRPrinter.print_stmt(stmt);
        }
    }
}
