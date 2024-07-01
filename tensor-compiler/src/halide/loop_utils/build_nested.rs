use crate::{ halide::{ for_stmt::For, stmt::Stmt }, iter_val::IterVar };

pub fn build_nested_for<T: Into<Stmt>>(iter_vars: &[IterVar], main_stmt: T) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(idx: usize, iter_vars: &[IterVar], main_stmt: T) -> Stmt {
        if idx == iter_vars.len() {
            main_stmt.into()
        } else {
            let to_add = For::make(
                iter_vars[idx].var(),
                iter_vars[idx].start(),
                iter_vars[idx].end(),
                build_recursive(idx + 1, iter_vars, main_stmt)
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, iter_vars, main_stmt)
}
