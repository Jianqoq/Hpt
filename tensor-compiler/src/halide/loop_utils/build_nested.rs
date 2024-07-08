use crate::{
    halide::{ for_stmt::For, seq_stmt::Seq, stmt::Stmt },
    hlir::schedule::new_iter::{ Node, RcMut, Stage },
    iter_var::IterVar,
};

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

pub fn build_nested_for2<T: Into<Stmt>>(iter_vars: &[RcMut<Node>], main_stmt: T) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(idx: usize, iter_vars: &[RcMut<Node>], main_stmt: T) -> Stmt {
        if idx == iter_vars.len() {
            main_stmt.into()
        } else {
            let to_add = For::make(
                iter_vars[idx].borrow().var(),
                iter_vars[idx].borrow().start(),
                iter_vars[idx].borrow().end(),
                build_recursive(idx + 1, iter_vars, main_stmt)
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, iter_vars, main_stmt)
}

#[rustfmt::skip]
pub fn build_nested_for3(stage: &Stage, mut main_stmt: Stmt) -> Stmt {
    let mut axes = stage.leaf_id
        .borrow()
        .iter()
        .map(|(node_ptr, id)| { (stage.address_map.borrow()[&*node_ptr].clone(), *id) })
        .collect::<Vec<_>>();
    axes.sort_by(|a, b| a.1.cmp(&b.1));
    let axes = axes
        .iter()
        .map(|(node, _)| node.clone())
        .collect::<Vec<_>>();
    let mut fors = None;
    for i in axes.iter() {
        if let Some(stages) = stage.attached_stage.borrow().get(&(i.as_ptr() as usize)) {
            for i in stages.iter() {
                main_stmt = build_nested_for3(&*i.borrow(), main_stmt);
            }
            if fors.is_none() {
                fors = Some(For::make(i.borrow().var(), i.borrow().start(), i.borrow().end(), Stmt::None));
            } else {
                fors = Some(For::make(i.borrow().var(), i.borrow().start(), i.borrow().end(), Stmt::For(fors.unwrap())));
            }
        } else {
            if fors.is_none() {
                fors = Some(For::make(i.borrow().var(), i.borrow().start(), i.borrow().end(), Stmt::None));
            } else {
                fors = Some(For::make(i.borrow().var(), i.borrow().start(), i.borrow().end(), Stmt::For(fors.unwrap())));
            }
        }
    }
    if let Some(fors) = fors {
        main_stmt = Stmt::For(fors);
    } else {
        main_stmt = Stmt::None;
    }
    main_stmt
}
