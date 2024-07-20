use std::sync::Arc;

use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::{ Int, Load, Malloc, Sub },
    for_stmt::For,
    inplace_store_stmt::InplaceMul,
    let_stmt::LetStmt,
    module::{ Function, FunctionType },
    primitive_type::{ PrimitiveType, Ptr },
    return_stmt::ReturnStmt,
    seq_stmt::Seq,
    stmt::Stmt,
    store_stmt::StoreStmt,
    variable::Variable,
};

pub fn declare_shape_to_strides() -> Function {
    let strides = LetStmt::make(
        &Variable::make("strides"),
        Malloc::make(Dtype::I64, Variable::make("ndim")),
        Stmt::None
    );
    let size = LetStmt::make(&Variable::make("size"), Int::make(Dtype::I64, 1), Stmt::None);
    let mut for_stmt = For::make(
        &Variable::make("i"),
        Sub::make(Variable::make("shape"), 1i64),
        0i64,
        -1i64,
        Stmt::None
    );
    let tmp = LetStmt::make(
        &Variable::make("tmp"),
        Load::make("shape", Variable::make("i")),
        Stmt::None
    );
    let store = StoreStmt::make(
        &Variable::make("strides"),
        Variable::make("i"),
        Variable::make("size")
    );
    let new_size = InplaceMul::make(Variable::make("size"), Variable::make("tmp"));
    let seq = Stmt::Seq(Seq::make(vec![Stmt::LetStmt(tmp), store.into(), new_size.into()]));
    for_stmt.set_stmt(seq);
    let ret = Stmt::Return(ReturnStmt::make(vec![Variable::make("strides").into()]));
    let body = Stmt::Seq(
        Seq::make(vec![Stmt::LetStmt(strides), size.into(), for_stmt.into(), ret])
    );
    let arg_ty = PrimitiveType::Ptr(Ptr {
        inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
    });
    Function {
        ty: FunctionType::new(
            PrimitiveType::Ptr(Ptr {
                inner: Arc::new(PrimitiveType::Dtype(Dtype::I64)),
            }),
            vec![(format!("shape"), arg_ty), (format!("ndim"), PrimitiveType::Dtype(Dtype::I64))]
        ),
        name: "shape_to_strides".to_string().into(),
        body,
    }
}

// pub fn predict_broadcast_shape(a_shape: &[i64], b_shape: &[i64]) -> anyhow::Result<Shape> {
//     let (longer, shorter) = if a_shape.len() >= b_shape.len() {
//         (a_shape, b_shape)
//     } else {
//         (b_shape, a_shape)
//     };

//     let padded_shorter = try_pad_shape(shorter, longer.len());
//     let mut result_shape = vec![0; longer.len()];

//     for (i, (&longer_dim, &shorter_dim)) in longer.iter().zip(&padded_shorter).enumerate() {
//         result_shape[i] = if longer_dim == shorter_dim || shorter_dim == 1 {
//             longer_dim
//         } else if longer_dim == 1 {
//             shorter_dim
//         } else {
//             return Err(
//                 ErrHandler::BroadcastError(
//                     format!(
//                         "Cannot broadcast shape: {:?} to {:?}, at axis {}.",
//                         a_shape,
//                         b_shape,
//                         i
//                     )
//                 ).into()
//             );
//         };
//     }

//     Ok(Shape::from(result_shape))
// }

pub fn predict_broadcast_shape() -> Function {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_declare_shape_to_strides() {
        let func = declare_shape_to_strides();
        let expected =
            r#"fn shape_to_strides(shape: *i64, ndim: i64) -> *i64 {
    let strides = malloc<i64>(ndim);
    let size = 1;
    for i in range(shape - 1, 0, -1) {
        let tmp = shape[i];
        strides[i] = size;
        size *= tmp;
    }
    return strides;
}"#;
        assert_eq!(format!("{}", func), expected);
    }
}
