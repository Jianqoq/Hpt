#![allow(unused_imports)]
use hashbrown::HashMap;
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
    let a = CmpNode::make(Shape::new([1, 8, 8]).into(), 0);
    let b = CmpNode::make(Shape::new([1]).into(), 1);
    let div = CmpNode::make_binop("div", a, b, 2);
    let mut comp = CmpNode::make_reduce(
        "max",
        &div,
        [Int::make(Dtype::I64, 2)],
        Int::make(Dtype::I64, 0).into(),
        3
    );
    comp.reshape(&Shape::new([1, 8, 1]));
    let sub = CmpNode::make_binop("sub", div, comp, 4);
    let exp = CmpNode::make_unop("exp", sub, 5);
    let mut sum = CmpNode::make_reduce("sum", &exp, [Int::make(Dtype::I64, 2)], Int::make(Dtype::I64, 0).into(), 6);
    sum.reshape(&Shape::new([1, 8, 1]));
    let div = CmpNode::make_binop("div", exp, sum, 7);
    let mut saved_exprs = HashMap::new();
    let exp_id = div.id();
    let expr = div.lower(true, &mut vec![], &mut saved_exprs);
    saved_exprs.insert(Variable::new(format!("%{}", exp_id)).into(), expr);
    for (k, v) in saved_exprs.iter() {
        println!("{}: {}", k, v);
    }
}
