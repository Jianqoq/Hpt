use hashbrown::HashMap;

use crate::{ halide::variable::Variable, hlir::exprs::ComputeNode };

pub struct FuseComputeNode {
    map: HashMap<Variable, ComputeNode>,
}
