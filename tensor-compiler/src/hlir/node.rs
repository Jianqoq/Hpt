#![allow(dead_code)]
use std::{ fmt::Display, sync::Arc };

use tensor_common::layout::Layout;
use tensor_types::dtype::Dtype;

use crate::op::Op;

use super::traits::{ HlirAcceptor, HlirVisitor };

pub struct Variable {
    name: Arc<String>,
}

impl Variable {
    pub fn make(name: Arc<String>) -> Variable {
        Variable { name }
    }
}

impl HlirAcceptor for Variable {
    fn accept<V: HlirVisitor>(&self, _visitor: &V) {}
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub struct Const {
    value: f64,
}

impl Const {
    pub fn make(value: f64) -> Const {
        Const { value }
    }
}

pub struct Tensor {
    layout: Layout,
    dtype: Dtype,
    error_msg: Option<Arc<String>>,
    op: Op,
    inputs: Arc<Vec<Node>>,
    outputs: Arc<Vec<Node>>,
    name: Option<Variable>,
}

impl Tensor {
    pub fn make(
        layout: Layout,
        dtype: Dtype,
        error_msg: Option<Arc<String>>,
        inputs: Arc<Vec<Node>>,
        outputs: Arc<Vec<Node>>,
        name: Option<Variable>,
        op: Op
    ) -> Tensor {
        Tensor { layout, dtype, error_msg, inputs, outputs, name, op }
    }
}

impl HlirAcceptor for Tensor {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        self.inputs.iter().for_each(|node| node.accept(visitor));
        self.outputs.iter().for_each(|node| node.accept(visitor));
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let id = &format!("{:p}", self);
        if let Some(name) = &self.name {
            write!(f, "{}", name)
        } else {
            write!(f, "{}", id)
        }
    }
}

pub struct Function {
    name: Arc<String>,
    inputs: Arc<Vec<Node>>,
    outputs: Arc<Vec<Node>>,
    body: Arc<Node>,
}

impl HlirAcceptor for Function {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        self.inputs.iter().for_each(|node| node.accept(visitor));
        self.outputs.iter().for_each(|node| node.accept(visitor));
        self.body.accept(visitor);
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "fn {} {{\n{}}}", self.name, self.body)
    }
}

pub struct IfThenElse {
    cond: Arc<Node>,
    then_case: Function,
    else_case: Function,
}

impl HlirAcceptor for IfThenElse {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        self.cond.accept(visitor);
        self.then_case.accept(visitor);
        self.else_case.accept(visitor);
    }
}

impl std::fmt::Display for IfThenElse {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "if {} {{\n{}}} else {{\n{}}}", self.cond, self.then_case, self.else_case)
    }
}

pub struct While {
    condition: Arc<Node>,
    body: Function,
}

impl HlirAcceptor for While {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        self.condition.accept(visitor);
        self.body.accept(visitor);
    }
}

impl std::fmt::Display for While {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "while {} {{\n{}}}", self.condition, self.body)
    }
}

pub struct For {
    var: Variable,
    start: i64,
    end: i64,
    body: Function,
}

impl For {
    pub fn make(var: Variable, start: i64, end: i64, body: Function) -> For {
        For { var, start, end, body }
    }
}

impl HlirAcceptor for For {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        self.var.accept(visitor);
        self.body.accept(visitor);
    }
}

impl std::fmt::Display for For {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "for {} in {}..{} {{\n{}}}", self.var, self.start, self.end, self.body)
    }
}

pub enum Node {
    Tensor(Tensor),
    Constant(Const),
    IfThenElse(IfThenElse),
    While(While),
    For(For),
    Function(Function),
}

impl HlirAcceptor for Node {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        match self {
            Node::Tensor(tensor) => tensor.accept(visitor),
            Node::Constant(_) => {}
            Node::IfThenElse(ite) => ite.accept(visitor),
            Node::While(while_) => while_.accept(visitor),
            Node::For(for_) => for_.accept(visitor),
            Node::Function(func) => func.accept(visitor),
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Node::Tensor(tensor) => write!(f, "{}", tensor),
            Node::Constant(const_) => write!(f, "{}", const_.value),
            Node::IfThenElse(ite) => write!(f, "{}", ite),
            Node::While(while_) => write!(f, "{}", while_),
            Node::For(for_) => write!(f, "{}", for_),
            Node::Function(func) => write!(f, "{}", func),
        }
    }
}
