#![allow(unused_imports)]
use tensor_common::block_manager::BlockType;
use tensor_compiler::{ front_end::tensor::Tensor, visualize };
pub fn custom_op(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let context = lhs.ctx().clone();
    context.borrow_mut().push_stack("custom_op", BlockType::Function);
    let result = lhs + rhs;
    let mut result = custom_op3(&result, &rhs);
    result = result * 2.0;
    context.borrow_mut().pop_stack();
    result
}

fn custom_op2(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let context = lhs.ctx().clone();
    context.borrow_mut().push_stack("custom_op2", BlockType::Function);
    let result = lhs / rhs;
    context.borrow_mut().pop_stack();
    result
}

fn custom_op4(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let context = lhs.ctx().clone();
    context.borrow_mut().push_stack("custom_op4", BlockType::Function);
    let result = lhs / rhs;
    context.borrow_mut().pop_stack();
    result
}

fn custom_op3(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let context = lhs.ctx().clone();
    context.borrow_mut().push_stack("custom_op3", BlockType::Function);
    let result = custom_op2(lhs, rhs);
    let result = custom_op4(&result, &result);
    context.borrow_mut().pop_stack();
    result
}

#[test]
pub fn test1() {
    visualize(|context| {
        let a = context.arange(0.0, 1024.0, 1.0);
        let a = a.reshape(&[1, 1024]);
        let b = context.arange(0.0, 1024.0, 1.0);
        let cond = context.cond(|| { &a + &b });
        let res = context
            .r#if(cond)
            .then(|| {
                let c = custom_op3(&a, &b);
                let t = &c + &a;
                [c, t]
            })
            .r#else(|| {
                let c = custom_op2(&a, &b);
                let t = &c + &b;
                [c, t]
            })
            .end();
        let c = &res[0] * 2.0;
        return [c];
    });
}
