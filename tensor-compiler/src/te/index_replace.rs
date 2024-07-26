use crate::halide::{substitute::subsititue_var::SubstituteVar, variable::Variable};

use super::stages::Body;


pub fn reduce_replace(origin_dim: usize, reduce_axes: &Vec<usize>, bodys: &mut Vec<Body>, input_id: usize, res_id: usize) {
    let mut subs_var = SubstituteVar::new();
    let mut cnt = 0;
    for idx in 0..origin_dim {
        if reduce_axes.contains(&idx) {
            subs_var.add_replacement(
                Variable::new(format!("ax{}", idx)),
                Variable::new(format!("{}red{}", res_id, idx))
            );
        } else {
            subs_var.add_replacement(
                Variable::new(format!("ax{}", idx)),
                Variable::new(format!("ax{}", cnt))
            );
            cnt += 1;
        }
    }
    subs_var.add_replacement(
        Variable::new(format!("%{}.s", input_id)),
        Variable::new(format!("%{}.s", res_id))
    );
    
    for i in bodys {
        i.replace_var(&mut subs_var);
    }
}