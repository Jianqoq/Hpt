use std::collections::HashMap;

use super::srg_node::SrgNode;



pub struct Executable {
    sorted: Vec<SrgNode>,
    outputs: Vec<SrgNode>,
}

impl Executable {
    pub fn execute(&self, map: HashMap<String, i64>) {
        for i in &self.outputs {
            let strides = (i.strides_cal)(&map);
        }
    }
}