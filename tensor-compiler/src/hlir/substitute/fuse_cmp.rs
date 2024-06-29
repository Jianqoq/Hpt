#![allow(dead_code)]
use hashbrown::HashMap;
use tensor_common::shape::Shape;

use crate::{
    halide::variable::Variable,
    hlir::{ expr::Expr, exprs::Tensor, traits::{ HlirMutateVisitor, MutatorGetSet } },
};

pub struct FuseComputeNode {
    map: HashMap<Variable, Tensor>,
    id: usize,
    expr: Expr,
}

impl MutatorGetSet for FuseComputeNode {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }
}

macro_rules! impl_binop_visitor {
    ($op: ident, $var: ident, $map: expr, $id: expr) => {
        let lhs = $op.lhs();
        let rhs = $op.rhs();
        let lhs: Variable = lhs.to_variable().unwrap().clone();
        let rhs: Variable = rhs.to_variable().unwrap().clone();
        if !$map.contains_key(&lhs) {
            panic!("add lhs {} not found in map", lhs);
        }
        if !$map.contains_key(&rhs) {
            panic!("add rhs {} not found in map", rhs);
        }
        let lhs = $map.get(&lhs).unwrap();
        let rhs = $map.get(&rhs).unwrap();
        let tensor = Tensor::make_binop(stringify!($op), lhs.clone(), rhs.clone(), $id);
        $map.insert($var.into(), tensor);
    };
}

impl HlirMutateVisitor for FuseComputeNode {
    fn visit_let(&mut self, let_: &crate::hlir::exprs::Let) {
        let var = let_.var();
        let value = let_.value();
        let body = let_.body();
        match value {
            Expr::Value(val) => {
                let mut tensor = Tensor::make(Shape::from([1]), val.dtype(), self.id);
                tensor.set_value(val);
                self.map.insert(var.into(), tensor);
            }
            Expr::Str(_) => todo!(),
            Expr::Variable(var) => {
                if let Some(tensor) = self.map.get(var) {
                    self.map.insert(var.into(), tensor.clone());
                } else {
                    panic!("variable {} not found in map", var);
                }
            }
            Expr::Tuple(_) => todo!(),
            Expr::Type(_) => unreachable!(),
            Expr::TensorType(_) => unreachable!(),
            Expr::OpNode(_) => unreachable!(),
            Expr::Tensor(tensor) => {
                self.map.insert(var.into(), tensor.clone());
            }
            Expr::Cast(cast) => {
                let val = cast.value();
                self.map.insert(
                    var.into(),
                    Tensor::make_const(cast.dtype(), val.casted_value(cast.dtype()), self.id)
                );
            }
            Expr::Add(add) => {
                impl_binop_visitor!(add, var, self.map, self.id);
            },
            Expr::Sub(_) => todo!(),
            Expr::Mul(_) => todo!(),
            Expr::Div(_) => todo!(),
            Expr::Mod(_) => todo!(),
            Expr::Min(_) => todo!(),
            Expr::Max(_) => todo!(),
            Expr::Eq(_) => todo!(),
            Expr::Ne(_) => todo!(),
            Expr::Lt(_) => todo!(),
            Expr::Le(_) => todo!(),
            Expr::Gt(_) => todo!(),
            Expr::Ge(_) => todo!(),
            Expr::And(_) => todo!(),
            Expr::Or(_) => todo!(),
            Expr::Xor(_) => todo!(),
            Expr::Not(_) => todo!(),
            Expr::Call(_) => todo!(),
            Expr::Select(_) => todo!(),
            Expr::Let(_) => todo!(),
            Expr::Alloc(_) => todo!(),
            Expr::If(_) => todo!(),
            Expr::For(_) => todo!(),
            Expr::While(_) => todo!(),
            Expr::Function(_) => todo!(),
            Expr::Slice(_) => todo!(),
            Expr::Return(_) => todo!(),
            Expr::None => todo!(),
        }
        self.id += 1;
        self.visit_expr(body);
    }

    fn visit_function(&mut self, func: &crate::hlir::exprs::Function) {
        let args = func.args();
        let body = func.body();
        for arg in args {
            if !self.map.contains_key(arg) {
                panic!("function argument {} not found in map", arg);
            }
        }
        self.visit_expr(body);
    }
}
