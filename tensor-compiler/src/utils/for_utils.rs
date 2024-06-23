use tensor_common::shape::Shape;

use crate::{exprs::Int, for_stmt::For, stmt::Stmt, r#type::{HalideirTypeCode, Type}, variable::Variable};


static INT_TYPE: Type = Type::new(HalideirTypeCode::Int, 64, 1);

pub fn build_nested_for<T: Into<Stmt>>(vars: &[Variable], shape: &Shape, main_stmt: T) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(idx: usize, vars: &[Variable], shape: &Shape, main_stmt: T) -> Stmt {
        if idx == vars.len() {
            main_stmt.into()
        } else {
            let var = &vars[idx];
            let to_add = For::make(
                var,
                Int::make(INT_TYPE, 0),
                Int::make(INT_TYPE, shape[idx] as i64),
                build_recursive(idx + 1, vars, shape, main_stmt),
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, vars, shape, main_stmt)
}
