use std::{ collections::{ HashMap, HashSet }, panic::Location, sync::Arc };

use tensor_types::{ dtype::Dtype, type_promote::NormalOut };

use crate::{
    halide::{ exprs::{ Add, Call, Int, Load, Reduce }, prime_expr::PrimeExpr, variable::Variable },
    hlir::tensor_slice::TensorLoad,
    iter_var::IterVar,
    to_prim_expr::ToPrimeExpr,
};

use super::{ operation::Operation, rc_mut::RcMut, srg::Srg, tensor::Tensor };

#[derive(Clone)]
pub struct Context {
    pub(crate) nodes: RcMut<HashMap<usize, Tensor>>,
    pub(crate) vars: RcMut<HashSet<Variable>>,
    pub(crate) id: RcMut<usize>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: RcMut::new(HashMap::new()),
            id: RcMut::new(0),
            vars: RcMut::new(HashSet::new()),
        }
    }

    #[track_caller]
    pub fn var(&mut self, name: &str) -> Variable {
        let var = Variable::new(name.to_string());
        self.vars.borrow_mut().insert(var.clone());
        var
    }

    #[track_caller]
    pub fn placeholder(&mut self, shape: &[&dyn ToPrimeExpr], dtype: Dtype) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let tensor = Tensor {
            shape: shape
                .into_iter()
                .map(|x| x.to_prime_expr())
                .collect::<Vec<PrimeExpr>>()
                .into(),
            body: (TensorLoad {
                var: Variable::new(format!("%{}", id)).into(),
                begins: (0..shape.len())
                    .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 0)))
                    .collect::<Vec<_>>()
                    .into(),
                axes: (0..shape.len())
                    .map(|x| Variable::new(format!("ax{}", x)).into())
                    .collect::<Vec<_>>()
                    .into(),
                steps: (0..shape.len())
                    .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 1)))
                    .collect::<Vec<_>>()
                    .into(),
                strides: (0..shape.len())
                    .map(|x| Load::make(Variable::new(format!("%{}.strides", id)), x).into())
                    .collect::<Vec<_>>()
                    .into(),
            }).into(),
            dtype,
            span: Location::caller(),
            op: Operation::None,
            inputs: Arc::new(vec![]),
            id,
        };
        self.nodes.borrow_mut().insert(id, tensor.clone());
        tensor
    }

    #[track_caller]
    pub fn reshape(&mut self, a: &Tensor, shape: &[&dyn ToPrimeExpr]) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let new_shape = shape
            .into_iter()
            .map(|x| x.to_prime_expr())
            .collect::<Vec<PrimeExpr>>();
        let tensor = Tensor {
            shape: new_shape.clone().into(),
            body: (TensorLoad {
                var: Variable::new(format!("%{}", a.id)).into(),
                begins: (0..new_shape.len())
                    .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 0)))
                    .collect::<Vec<_>>()
                    .into(),
                axes: (0..new_shape.len())
                    .map(|x| Variable::new(format!("ax{}", x)).into())
                    .collect::<Vec<_>>()
                    .into(),
                steps: (0..new_shape.len())
                    .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 1)))
                    .collect::<Vec<_>>()
                    .into(),
                strides: (0..new_shape.len())
                    .map(|x| Load::make(Variable::new(format!("%{}.strides", a.id)), x).into())
                    .collect::<Vec<_>>()
                    .into(),
            }).into(),
            dtype: a.dtype.clone(),
            span: Location::caller(),
            op: Operation::Reshape(new_shape.into()),
            inputs: Arc::new(vec![a.id]),
            id,
        };
        self.nodes.borrow_mut().insert(id, tensor.clone());
        tensor
    }

    #[track_caller]
    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        let lhs_shape = a.shape.clone();
        let rhs_shape = b.shape.clone();
        let mut res_shape = vec![];
        let (lhs_start, rhs_start) = if lhs_shape.len() < rhs_shape.len() {
            (0..rhs_shape.len() - lhs_shape.len()).for_each(|x| {
                res_shape.push(rhs_shape[x].clone());
            });
            (0, rhs_shape.len() - lhs_shape.len())
        } else if lhs_shape.len() > rhs_shape.len() {
            (0..lhs_shape.len() - rhs_shape.len()).for_each(|x| {
                res_shape.push(lhs_shape[x].clone());
            });
            (lhs_shape.len() - rhs_shape.len(), 0)
        } else {
            (0, 0)
        };
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
        lhs_shape[lhs_start..]
            .iter()
            .zip(rhs_shape[rhs_start..].iter())
            .for_each(|(x, y)| {
                if x == &one {
                    res_shape.push(y.clone());
                } else if y == &one {
                    res_shape.push(x.clone());
                } else if x == y {
                    res_shape.push(x.clone());
                } else {
                    panic!("Incompatible shapes. {} and {}", x, y);
                }
            });
        let a_load = TensorLoad {
            var: Variable::new(format!("%{}", a.id)).into(),
            begins: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 0)))
                .collect::<Vec<_>>()
                .into(),
            axes: (0..res_shape.len())
                .map(|x| Variable::new(format!("ax{}", x)).into())
                .collect::<Vec<_>>()
                .into(),
            steps: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 1)))
                .collect::<Vec<_>>()
                .into(),
            strides: (0..res_shape.len())
                .map(|x| Load::make(Variable::new(format!("%{}.strides", a.id)), x).into())
                .collect::<Vec<_>>()
                .into(),
        };
        let b_load = TensorLoad {
            var: Variable::new(format!("%{}", b.id)).into(),
            begins: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 0)))
                .collect::<Vec<_>>()
                .into(),
            axes: (0..res_shape.len())
                .map(|x| Variable::new(format!("ax{}", x)).into())
                .collect::<Vec<_>>()
                .into(),
            steps: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 1)))
                .collect::<Vec<_>>()
                .into(),
            strides: (0..res_shape.len())
                .map(|x| Load::make(Variable::new(format!("%{}.strides", b.id)), x).into())
                .collect::<Vec<_>>()
                .into(),
        };
        let add = PrimeExpr::Add(Add::make(a_load, b_load));
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let ret = Tensor {
            shape: res_shape.into(),
            body: add.into(),
            inputs: Arc::new(vec![a.id, b.id]),
            span: Location::caller(),
            id,
            dtype: a.dtype._add(b.dtype),
            op: Operation::Add,
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn sum(&mut self, a: &Tensor, init: &dyn ToPrimeExpr, axes: &[i64]) -> Tensor {
        let mut axes = axes.to_vec();
        for i in axes.iter_mut() {
            if *i < 0 {
                *i += a.shape.len() as i64;
            }
        }
        axes.sort();

        let mut res_shape = vec![];
        for i in 0..a.shape.len() as i64 {
            if !axes.contains(&i) {
                res_shape.push(a.shape[i as usize].clone());
            }
        }

        let a_load = TensorLoad {
            var: Variable::new(format!("%{}", a.id)).into(),
            begins: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 0)))
                .collect::<Vec<_>>()
                .into(),
            axes: (0..res_shape.len())
                .map(|x| Variable::new(format!("ax{}", x)).into())
                .collect::<Vec<_>>()
                .into(),
            steps: (0..res_shape.len())
                .map(|_| PrimeExpr::Int(Int::make(Dtype::I64, 1)))
                .collect::<Vec<_>>()
                .into(),
            strides: (0..res_shape.len())
                .map(|x| Load::make(Variable::new(format!("%{}.strides", a.id)), x).into())
                .collect::<Vec<_>>()
                .into(),
        };
        let reduce_iter_vars = axes
            .iter()
            .map(|x| {
                IterVar::new(0i64, a.shape[*x as usize].clone(), 1i64, &format!("red_{}", x))
            })
            .collect::<Vec<IterVar>>();
        let sum = PrimeExpr::Reduce(Reduce {
            identity: Arc::new(vec![init.to_prime_expr()]),
            iter_vars: Arc::new(reduce_iter_vars),
            expr: Arc::new(vec![a_load.into()]),
            op: "sum",
        });
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let ret = Tensor {
            shape: res_shape.into(),
            body: sum.into(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            op: Operation::Sum(
                Arc::new(
                    axes
                        .into_iter()
                        .map(|x| x as usize)
                        .collect()
                ),
                init.to_prime_expr(),
            ),
            dtype: a.dtype.clone(),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn sin(&mut self, a: &Tensor) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let sin = PrimeExpr::Call(Call::make("sin", &[Variable::new(format!("%{}", a.id))]));
        let ret = Tensor {
            shape: a.shape.clone(),
            body: sin.into(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            id,
            op: Operation::Sin,
            dtype: a.dtype.clone(),
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    #[track_caller]
    pub fn slice(
        &mut self,
        a: &Tensor,
        selections: &[
            (&dyn ToPrimeExpr /*begin */, &dyn ToPrimeExpr /*end */, &dyn ToPrimeExpr /*step */)
        ]
    ) -> Tensor {
        let id = self.id.borrow().clone();
        *self.id.borrow_mut() += 1;
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1i64));
        let zero = PrimeExpr::Int(Int::make(Dtype::I64, 0i64));
        let new_shape = selections
            .into_iter()
            .map(|(begin, end, step)| {
                let begin = begin.to_prime_expr();
                let end = end.to_prime_expr();
                let step = step.to_prime_expr();
                if &begin == &zero && &step == &one {
                    end.clone()
                } else if &step == &one {
                    end.clone() - begin.clone()
                } else {
                    (end.clone() - begin.clone()) / step.clone()
                }
            })
            .collect::<Vec<_>>();
        let axes = (0..new_shape.len())
            .map(|i| Variable::new(format!("ax{}", i)).into())
            .collect::<Vec<_>>();
        let tensor_load = TensorLoad::make(
            Variable::new(format!("%{}", a.id)),
            selections
                .into_iter()
                .map(|(begin, _, _)| begin.to_prime_expr())
                .collect::<Vec<PrimeExpr>>(),
            axes,
            selections
                .into_iter()
                .map(|(_, _, step)| step.to_prime_expr())
                .collect::<Vec<PrimeExpr>>(),
            (0..new_shape.len())
                .map(|x| Load::make(Variable::new(format!("%{}.strides", a.id)), x).into())
                .collect::<Vec<PrimeExpr>>()
        );
        let ret = Tensor {
            shape: new_shape.into(),
            body: tensor_load.into(),
            inputs: Arc::new(vec![a.id]),
            span: Location::caller(),
            op: Operation::Slice(
                selections
                    .into_iter()
                    .map(|(x, y, z)| { (x.to_prime_expr(), y.to_prime_expr(), z.to_prime_expr()) })
                    .collect::<Vec<_>>()
                    .into()
            ),
            dtype: a.dtype.clone(),
            id,
        };
        self.nodes.borrow_mut().insert(id, ret.clone());
        ret
    }

    pub fn to_srg(self) -> HashMap<usize, Srg> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::halide::printer::IRPrinter;

    use super::*;

    #[test]
    fn test_placeholder() {
        let mut context = Context {
            nodes: RcMut::new(HashMap::new()),
            id: RcMut::new(0),
            vars: RcMut::new(HashSet::new()),
        };
        let tensor = context.placeholder(&[&10i64], Dtype::F32);
        IRPrinter.print_expr(tensor.body)
    }

    #[test]
    fn test_slice() {
        let mut context = Context {
            nodes: RcMut::new(HashMap::new()),
            id: RcMut::new(0),
            vars: RcMut::new(HashSet::new()),
        };
        let tensor = context.placeholder(&[&10i64], Dtype::F32);
        let tensor = context.slice(
            &tensor,
            &[
                (&1i64, &5i64, &1i64),
                (&2i64, &8i64, &1i64),
            ]
        );
        IRPrinter.print_expr(tensor.body)
    }

    #[test]
    fn test_add() {
        let mut context = Context {
            nodes: RcMut::new(HashMap::new()),
            id: RcMut::new(0),
            vars: RcMut::new(HashSet::new()),
        };
        let a = context.placeholder(&[&10i64], Dtype::F32);
        let b = context.placeholder(&[&10i64], Dtype::F32);
        let c = context.add(&a, &b);
        IRPrinter.print_expr(c.body)
    }
}
