use crate::halide::{substitute::subsititue_var::SubstituteVar, variable::Variable};

use super::stages::Body;


pub fn reduce_replace(origin_dim: usize, reduce_axes: &Vec<usize>, bodys: &mut Vec<Body>, input_id: usize, res_id: usize) {
    let mut subs_var = SubstituteVar::new();
    let mut cnt = 0;

    let start = origin_dim - reduce_axes.len();

    for (idx, i) in (start..origin_dim).enumerate() {
        subs_var.add_replacement(
            Variable::new(format!("ax{}", i)),
            Variable::new(format!("{}red{}", res_id, reduce_axes[idx]))
        );
    }

    for i in 0..start {
        subs_var.add_replacement(
            Variable::new(format!("ax{}", i)),
            Variable::new(format!("ax{}", cnt))
        );
        cnt += 1;
    }
    subs_var.add_replacement(
        Variable::new(format!("%{}.s", input_id)),
        Variable::new(format!("%{}.s", res_id))
    );
    
    for i in bodys {
        i.replace_var(&mut subs_var);
    }
}