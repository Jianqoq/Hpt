#![allow(unused_imports)]
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use super::{ expr::Expr, exprs::*, func_type::Type, printer::HlirPrinter };

#[test]
fn test_build_main() {
    let mut main = Function::make("main", &[], &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let let_ = Let::make("b", &a, Expr::None);
    main.set_body(let_);
    HlirPrinter.print(main)
}

#[test]
fn test_for() {
    let mut main = Function::make("main", &[], &Type::make_none(), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let start = Value::make(Dtype::I32, 0);
    let end = Value::make(Dtype::I32, 10);
    let step = Value::make(Dtype::I32, 1);

    let slice = Slice::make("b", [
        (Value::make(Dtype::I32, 0), Value::make(Dtype::I32, 0), Value::make(Dtype::I32, 0)),
    ]);
    let for_ = For::make("i", start, end, step, slice);

    let b = Let::make("b", &a, for_);

    main.set_body(b);
    HlirPrinter.print(main)
}
