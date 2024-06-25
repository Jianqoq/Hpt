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

pub fn test1() -> serde_json::Value {
    let graph = visualize(|context| {
        let a = context.arange(0.0, 1024.0, 1.0);
        let a = a.reshape(&[1, 1024]);
        let b = context.arange(0.0, 1024.0, 1.0);
        let cond = context.cond(|| { &a + &b });
        let res = context
            .r#if(cond)
            .then(|| {
                let c = custom_op3(&a, &b);
                c
            })
            .r#else(|| {
                let c = custom_op2(&a, &b);
                c
            })
            .end();
        let c = &res * 2.0;
        return [c];
    });
    graph
}

// pub fn test2() -> serde_json::Value {
//     let graph = visualize(|context| {
//         let a = context.arange(0.0, 1024.0, 1.0);
//         let a = a.reshape(&[1, 1024]);
//         let b = context.arange(0.0, 1024.0, 1.0);
//         let c = custom_op3(&a, &b);
//         let d = custom_op4(&c, &c);
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [d];
//     });
//     serde_json::json!(graph)
// }

// pub fn test3() -> serde_json::Value {
//     let graph = visualize(|context| {
//         let a = context.arange(0.0, 1024.0, 1.0);
//         let a = a.reshape(&[1, 1024]);
//         let b = context.arange(0.0, 1024.0, 1.0);
//         let c = custom_op3(&a, &b);
//         let r = &c + &b;
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [r];
//     });
//     serde_json::json!(graph)
// }

// pub fn test4() -> serde_json::Value {
//     let graph = visualize(|context| {
//         let a = context.arange(0.0, 1024.0, 1.0);
//         let a = a.reshape(&[1, 1024]);
//         let b = context.arange(0.0, 1024.0, 1.0);
//         let r = &a + &b;
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [r];
//     });
//     serde_json::json!(graph)
// }

// pub fn conseq_uary_fuse() -> serde_json::Value {
//     let graph = visualize_mir_fuse(|context| {
//         let a = context.randn::<f64, _>(&[1, 8]);
//         let b = context.randn::<f64, _>(&[1, 8]);
//         let r = &a.sin().cos().abs().acos().tan() + &b.abs().acos().tan();
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [r];
//     });
//     serde_json::json!(graph)
// }

// pub fn converge() -> serde_json::Value {
//     let graph = visualize_mir_fuse(|context| {
//         let a = context.randn::<f64, _>(&[1, 8]);
//         let b = context.randn::<f64, _>(&[1, 8]);
//         let c = a.sin() * b.sin();
//         let d = c.sum(&[1], true) + c.sin();
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [d];
//     });
//     serde_json::json!(graph)
// }

// pub fn double_converge() -> serde_json::Value {
//     let graph = visualize_mir_fuse(|context| {
//         let a = context.randn::<f64, _>(&[1, 8]);
//         let b = context.randn::<f64, _>(&[1, 8]);
//         let c = a.cos() + b.sin();
//         let d = c.sin() + c.cos();
//         let e = d.sum(&[1], true) * d.cos();
//         context.set_inputs(vec![a.id(), b.id()]);
//         return [e];
//     });
//     serde_json::json!(graph)
// }
