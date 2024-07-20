use tensor_types::dtype::Dtype;

use crate::halide::{ exprs::Int, let_stmt::LetStmt, module::Function, prime_expr::PrimeExpr, variable::Variable };


// pub fn shape_to_strides(shape: &PrimeExpr) -> PrimeExpr {
//     let mut strides = vec![PrimeExpr::Int(Int::make(Dtype::I64, 0)); shape.len()];
//     let mut size = PrimeExpr::Int(Int::make(Dtype::I64, 1));
//     for i in (0..shape.len()).rev() {
//         let tmp = shape[i];
//         strides[i] = size as i64;
//         size *= tmp;
//     }
//     strides.into()
// }

// pub fn declare_shape_to_strides() -> Function {
//     let body = LetStmt::make(&Variable::make("strides"), , body)
//     todo!()
// }