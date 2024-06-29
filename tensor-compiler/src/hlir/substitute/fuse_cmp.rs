#![allow(dead_code)]
use hashbrown::HashMap;
use tensor_common::shape::Shape;

use crate::{
    halide::variable::Variable,
    hlir::{ expr::Expr, exprs::Tensor, traits::{ HlirMutateVisitor, MutatorGetSet } },
};

pub struct FuseComputeNode {
    map: HashMap<Expr, Tensor>,
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
            Expr::Variable(_) => todo!(),
            Expr::Tuple(_) => todo!(),
            Expr::Type(_) => todo!(),
            Expr::TensorType(_) => todo!(),
            Expr::OpNode(_) => todo!(),
            Expr::Tensor(_) => todo!(),
            Expr::Cast(_) => todo!(),
            Expr::Add(_) => todo!(),
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
            let expr: Expr = arg.into();
            if !self.map.contains_key(&expr) {
                panic!("function argument {} not found in map", arg);
            }
        }
        self.visit_expr(body);
    }
}
