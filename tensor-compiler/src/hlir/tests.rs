#![allow(unused_imports)]
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use super::{ exprs::{ Function, Let, Tensor, Value }, node::Expr, printer::HlirPrinter };

#[test]
fn test_build_main() {
    let mut main = Function::make("main", &[], Value::make(Dtype::I32, 1), Expr::None);

    let a = Tensor::make("a", Shape::new([1, 2, 3]).into(), Dtype::BF16);

    let let_ = Let::make("b", &a);
    main.set_body(let_);
    HlirPrinter.print(main)
}
