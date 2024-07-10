use crate::{
    halide::{ for_stmt::For, seq_stmt::Seq, stmt::Stmt },
    hlir::schedule::schedule::{ Node, RcMut, Stage },
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
                iter_vars[idx].step(),
                build_recursive(idx + 1, iter_vars, main_stmt)
            );
            Stmt::For(to_add)
        }
    }

    build_recursive(0, iter_vars, main_stmt)
}

pub fn build_nested_for_helper<T: Into<Stmt>>(
    stage: RcMut<Stage>,
    iter_vars: &[RcMut<Node>],
    main_stmt: T
) -> Stmt {
    fn build_recursive<T: Into<Stmt>>(
        idx: usize,
        stage: RcMut<Stage>,
        iter_vars: &[RcMut<Node>],
        main_stmt: T
    ) -> Stmt {
        if idx == iter_vars.len() {
            main_stmt.into()
        } else {
            let mut seq = None;
            if
                let Some(stages) = stage
                    .borrow()
                    .attached_stage.borrow()
                    .get(&(iter_vars[idx].as_ptr() as usize))
            {
                let mut vec = Vec::with_capacity(stages.len());
                for i in stages {
                    vec.push(i.borrow().to_halid());
                }
                seq = Some(vec);
            }
            let to_add = if let Some(mut seq) = seq {
                seq.push(build_recursive(idx + 1, stage, iter_vars, main_stmt));
                For::make(
                    iter_vars[idx].borrow().var(),
                    iter_vars[idx].borrow().start(),
                    iter_vars[idx].borrow().end(),
                    iter_vars[idx].borrow().step(),
                    Seq::make(seq)
                )
            } else {
                For::make(
                    iter_vars[idx].borrow().var(),
                    iter_vars[idx].borrow().start(),
                    iter_vars[idx].borrow().end(),
                    iter_vars[idx].borrow().step(),
                    build_recursive(idx + 1, stage, iter_vars, main_stmt)
                )
            };
            Stmt::For(to_add)
        }
    }

    build_recursive(0, stage, iter_vars, main_stmt)
}

#[rustfmt::skip]
pub fn build_nested_for2(stage: RcMut<Stage>, main_stmt: Stmt) -> Stmt {
    let mut axes = stage.borrow().leaf_id
        .borrow()
        .iter()
        .map(|(node_ptr, id)| { (stage.borrow().address_map.borrow()[&*node_ptr].clone(), *id) })
        .collect::<Vec<_>>();
    axes.sort_by(|a, b| a.1.cmp(&b.1));
    let axes = axes
        .iter()
        .map(|(node, _)| node.clone())
        .collect::<Vec<_>>();
    build_nested_for_helper(stage, &axes, main_stmt)
}
