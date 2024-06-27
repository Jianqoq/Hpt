use tensor_common::shape::Shape;
use tensor_types::dtype::Dtype;

use crate::halide::{
    exprs::Int,
    for_stmt::For,
    stmt::Stmt,
    variable::Variable,
};

pub fn build_nested_for<T: Into<Stmt>>(vars: &[Variable], shape: &Shape, main_stmt: T) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(
        idx: usize,
        vars: &[Variable],
        shape: &Shape,
        main_stmt: T
    ) -> Stmt {
        if idx == vars.len() {
            main_stmt.into()
        } else {
            let var = &vars[idx];
            let to_add = For::make(
                var,
                Int::make(Dtype::I64, 0),
                Int::make(Dtype::I64, shape[idx] as i64),
                build_recursive(idx + 1, vars, shape, main_stmt)
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, vars, shape, main_stmt)
}
