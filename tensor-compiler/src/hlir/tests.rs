#![allow(unused_imports)]
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ Int, Load }, printer::IRPrinter, variable::Variable },
    registry::{ Closures, MANAGER },
};

use super::{ expr::Expr, exprs::*, func_type::Type, printer::HlirPrinter };

#[test]
fn test_build_main() {
    let args: [Variable; 0] = [];
    let mut main = Function::make("main", &args, &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let let_ = Let::make("b", &a, Expr::None);
    main.set_body(let_);
    HlirPrinter.print(main)
}

// this should panic, for loop should always return value
#[test]
fn test_for_fail() {
    let args: [Variable; 0] = [];
    let mut main = Function::make("main", &args, &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let start = Value::make(Dtype::I32, 0);
    let end = Value::make(Dtype::I32, 10);
    let step = Value::make(Dtype::I32, 1);

    let for_var = Variable::make("i");
    let slice = Slice::make("b", [
        (Value::make(Dtype::I32, 0), &for_var, Value::make(Dtype::I32, 2)),
    ]);
    let let_ = Let::make("d", &slice, Expr::None);
    let for_ = For::make(for_var, start, end, step, let_);

    let b = Let::make("b", &a, for_);

    main.set_body(b);
    HlirPrinter.print(main)
}

// this should work
#[test]
fn test_for() {
    let args: [Variable; 0] = [];
    let mut main = Function::make("main", &args, &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let start = Value::make(Dtype::I32, 0);
    let end = Value::make(Dtype::I32, 10);
    let step = Value::make(Dtype::I32, 1);

    let for_var = Variable::make("i");

    let slice = Slice::make("b", [
        (Value::make(Dtype::I32, 0), &for_var, Value::make(Dtype::I32, 2)),
    ]);
    let let_ = Let::make("d", &slice, Expr::None);
    let for_ = For::make(for_var, start, end, step, let_);

    let ret = Let::make("ret", &for_, Expr::None);

    let b = Let::make("b", &a, ret);

    main.set_body(b);
    HlirPrinter.print(main)
}

#[test]
fn test_fusion() {
    let a = Tensor::make("a", Shape::new([1, 8, 8]).into(), Dtype::BF16);
    let b = Tensor::make("b", Shape::new([1]).into(), Dtype::BF16);
    let div_op = MANAGER.lock().unwrap().get("div").cloned().unwrap();
    let a_load = Load::make(
        a.name(),
        [Int::make(Dtype::I32, 1), Int::make(Dtype::I32, 2), Int::make(Dtype::I32, 3)]
    );
    let b_load = Load::make(b.name(), [Int::make(Dtype::I32, 0)]);
    let expr = ComputeNode::make_binop(div_op, a_load, b_load);
    println!("{}", expr);
}
