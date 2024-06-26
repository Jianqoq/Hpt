#![allow(unused_imports)]
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use super::{
    expr::Expr,
    exprs::{ Function, Let, Tensor, Value },
    func_type::Type,
    printer::HlirPrinter,
};

#[test]
fn test_build_main() {
    let mut main = Function::make("main", &[], &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let let_ = Let::make("b", &a);
    main.set_body(let_);
    HlirPrinter.print(main)
}

#[test]
fn test_for() {
    let mut main = Function::make("main", &[], &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let b = Let::make("b", &a);
    main.set_body(b);
    HlirPrinter.print(main)
}