use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use super::{ hstrides::HStrides, tensor::Tensor };

pub struct Executable {
    strides_vec: (*mut (*mut i64, usize), usize),
    var_map: (*mut i64, usize),
}

impl Executable {
    pub fn new(
        var_map: HashMap<String, i64>,
        nodes: &HashMap<usize, Tensor>,
        strides_cal: Arc<dyn Fn(&HashMap<String, i64>) -> Vec<HStrides>>
    ) -> Self {
        let strides = strides_cal(&var_map);
        let strides_vec = unsafe {
            let strides_vec = std::alloc::alloc(
                std::alloc::Layout
                    ::from_size_align(strides.len() * std::mem::size_of::<(*mut i64, usize)>(), 8)
                    .unwrap()
            ) as *mut (*mut i64, usize);
            for (idx, s) in strides.iter().enumerate() {
                strides_vec.add(idx).write((s.to_aligned_ptr(), s.strides.len()));
            }
            (strides_vec, strides.len())
        };
        let mut vars = vec![];
        let mut visited = HashSet::new();
        for node in nodes.values() {
            for i in node.shape.iter() {
                if let Some(var) = i.to_variable() {
                    if visited.contains(var) {
                        continue;
                    } else {
                        vars.push(var.clone());
                        visited.insert(var);
                    }
                }
            }
        }
        vars.sort();
        let shape_vars = unsafe {
            let shape_vars = std::alloc::alloc(
                std::alloc::Layout
                    ::from_size_align(vars.len() * std::mem::size_of::<i64>(), 8)
                    .unwrap()
            ) as *mut i64;
            for (idx, var) in vars.iter().enumerate() {
                shape_vars.add(idx).write(var_map[var.name()]);
            }
            (shape_vars, vars.len())
        };

        Self {
            strides_vec,
            var_map: shape_vars,
        }
    }
    pub fn execute(&self) {
        unsafe {}
    }
}
