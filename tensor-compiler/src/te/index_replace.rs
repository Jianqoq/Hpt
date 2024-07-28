use crate::halide::{substitute::subsititue_var::SubstituteVar, variable::Variable};

use super::{stages::Body, transpose_axes::TransposeAxes};


pub fn reduce_replace(origin_dim: usize, reduce_axes: &Vec<usize>, bodys: &mut Vec<Body>, input_id: usize, res_id: usize) {

    let mut transpose_axes = vec![];

    for i in 0..origin_dim {
        if !reduce_axes.contains(&i) {
            transpose_axes.push(i);
        }
    }
    transpose_axes.extend(reduce_axes.clone());

    let mut transpose_axes = TransposeAxes::new(transpose_axes);

    for i in bodys.iter_mut() {
        i.accept_mutate(&mut transpose_axes);
    }

    let mut subs_var = SubstituteVar::new();
    let mut cnt = 0;

    for i in 0..origin_dim {
        if !reduce_axes.contains(&i) {
            subs_var.add_replacement(
                Variable::new(format!("ax{}", i)),
                Variable::new(format!("ax{}", cnt))
            );
            cnt += 1;
        } else {
            subs_var.add_replacement(
                Variable::new(format!("ax{}", i)),
                Variable::new(format!("{}red{}", res_id, i))
            );
        }
    }
    subs_var.add_replacement(
        Variable::new(format!("%{}.s", input_id)),
        Variable::new(format!("%{}.s", res_id))
    );
    
    for i in bodys {
        i.accept_mutate(&mut subs_var);
    }
}