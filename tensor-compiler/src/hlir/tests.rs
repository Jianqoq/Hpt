#![allow(unused_imports)]
use tensor_common::layout::Layout;
use tensor_types::dtype::Dtype;

use super::{ exprs::{ Function, Tensor, Value }, node::Expr, printer::HlirPrinter };

#[test]
fn test_build_main() {
    let main = Function::make("main", &[], Value::make(Dtype::I32, 1), Expr::None);
    // let layout = Layout::new(shape, strides)
    // let a = Tensor::make("a", , dtype)
    HlirPrinter.print(main)
}
