use std::collections::HashMap;

use super::srg_node::SrgNode;



pub struct Executable {
    sorted: Vec<SrgNode>,
    var_map: HashMap<String, i64>,
}

impl Executable {
    pub fn execute(&self, map: HashMap<String, i64>) {
    }
}