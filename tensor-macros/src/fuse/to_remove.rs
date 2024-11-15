use std::collections::HashSet;

use super::{ fuse::FusionGroup, gen_fuse::GenFuse };

pub(crate) struct ToRemove {
    pub(crate) to_remove: Vec<HashSet<syn::Ident>>,
    pub(crate) next: Option<Box<ToRemove>>,
}

pub(crate) fn gen_to_remove(gen_fuse: &GenFuse, fuse_group: &FusionGroup) -> ToRemove {
    let to_remove = _gen_to_remove(&fuse_group.vars, &gen_fuse.fused_inputs, &gen_fuse.fused_outs);
    let mut ret = ToRemove { to_remove, next: None };
    match (&gen_fuse.next_gen_fuse, &fuse_group._next_group) {
        (Some(next_gen_fuse), Some(next_fuse_group)) => {
            ret.next = Some(Box::new(gen_to_remove(&next_gen_fuse, &next_fuse_group)));
        }
        _ => {}
    }
    ret
}

pub(crate) fn _gen_to_remove(
    fused_group: &Vec<HashSet<syn::Ident>>,
    fused_inputs: &Vec<HashSet<syn::Ident>>,
    fused_outs: &Vec<syn::Ident>
) -> Vec<HashSet<syn::Ident>> {
    let mut to_remove = Vec::new();
    for ((input, total), out) in fused_inputs
        .iter()
        .zip(fused_group.iter())
        .zip(fused_outs.iter()) {
        let mut intermediate = total
            .iter()
            .map(|i| i.clone())
            .collect::<HashSet<_>>();
        for input in input {
            intermediate.remove(input);
        }
        intermediate.remove(out);
        to_remove.push(intermediate);
    }
    to_remove
}
