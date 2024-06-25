#![allow(dead_code)]
use std::rc::Rc;

use tensor_common::block_manager::BlockType;

use crate::op::Op;

use super::{ context::Context, tensor::Tensor };

pub trait ControlFlowState {
    fn block_id(&self) -> usize;
    fn push_stack(&self, name: &str, block_type: BlockType);
    fn pop_stack(&self);
}

pub struct BranchResult {
    res: Vec<Tensor>,
}

pub struct If<'a, T> {
    condition: T,
    context: &'a Context,
}

pub struct Then<'a, R> {
    context: &'a Context,
    ret: R,
}

pub struct Else<'a, R> {
    context: &'a Context,
    ret: Vec<R>,
}

pub struct While<'a> {
    cond_block_id: usize,
    context: &'a Context,
}

impl<'a, T> If<'a, T> {
    pub fn new(condition: T, context: &'a Context) -> Self {
        context.push_stack("if", BlockType::If);
        If { condition, context }
    }

    pub fn then<F, R>(self, block: F) -> Then<'a, R> where F: FnOnce() -> R {
        self.context.push_stack("if_then", BlockType::Function);
        let ret = block();
        self.context.pop_stack();
        Then {
            context: self.context,
            ret,
        }
    }
}

impl<'a, R> Then<'a, R> {
    pub fn r#else<F>(self, block: F) -> Else<'a, R> where F: FnOnce() -> R {
        self.context.push_stack("then_else", BlockType::Function);
        let ret = block();
        self.context.pop_stack();
        Else {
            context: self.context,
            ret: vec![self.ret, ret],
        }
    }
}

impl<'a, R> Else<'a, R> where R: Merge<Output = R> {
    pub fn end(self) -> R {
        self.context.pop_stack();
        let ret = self.ret.into_iter().fold(R::init(self.context), |acc, r| acc.merge(r));
        ret.update_ctx(self.context);
        ret
    }
}

impl<'a> While<'a> {
    pub fn new(context: &'a Context) -> Self {
        let cond_block_id = context.block_id();
        While {
            cond_block_id,
            context,
        }
    }

    pub fn body<F>(self, block: F) -> Self where F: FnOnce() {
        self.context.push_stack("while_body", BlockType::Function);
        block();
        self.context.pop_stack();
        self
    }
}

pub struct Condition {
    then: usize,
    else_: usize,
}

impl Condition {
    pub fn new(then: usize, else_: usize) -> Self {
        Condition { then, else_ }
    }
}

impl Into<BranchResult> for Tensor {
    fn into(self) -> BranchResult {
        BranchResult { res: vec![self] }
    }
}

impl<const N: usize> From<[Tensor; N]> for BranchResult {
    fn from(tensors: [Tensor; N]) -> Self {
        BranchResult { res: tensors.to_vec() }
    }
}

pub trait Merge {
    type Output;
    fn merge(self, other: Self) -> Self::Output;
    fn init(ctx: &Context) -> Self::Output;
    fn update_ctx(&self, ctx: &Context);
}

impl<const N: usize> Merge for [Tensor; N] {
    type Output = [Tensor; N];
    fn merge(self, other: Self) -> Self::Output {
        let mut res = self.clone();
        self.into_iter()
            .zip(other.into_iter())
            .enumerate()
            .for_each(|(i, (a, b))| {
                res[i] = a.merge(b);
            });
        res
    }
    fn init(ctx: &Context) -> Self::Output {
        std::array::from_fn(|_| {
            let mut t: Tensor = ctx.into();
            *t.op_mut() = Op::Merge;
            t
        })
    }
    fn update_ctx(&self, ctx: &Context) {
        for t in self.iter() {
            let mut inner = ctx.ctx().borrow_mut();
            inner.nodes_mut().insert(t.id, t.clone().into());
        }
    }
}

impl Merge for Tensor {
    type Output = Tensor;
    fn merge(mut self, other: Self) -> Self::Output {
        Rc::make_mut(&mut self.inputs).push(other.id);
        *self.layout_mut() = other.layout.clone();
        self
    }
    fn init(ctx: &Context) -> Self::Output {
        let mut res: Tensor = ctx.into();
        *res.op_mut() = Op::Merge;
        res
    }
    fn update_ctx(&self, ctx: &Context) {
        let mut inner = ctx.ctx().borrow_mut();
        inner.nodes_mut().insert(self.id, self.clone().into());
    }
}
