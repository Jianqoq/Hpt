use std::sync::Arc;

use hashbrown::HashMap;
use lazy_static::lazy_static;

use crate::hlir::{ expr::Expr, exprs::{ Call, OpNode } };

pub struct Registry<F> {
    name: String,
    f: F,
}

pub struct OpRegEntry {
    name: String,
    op_node: Arc<OpNode>,
}

pub struct AttrRegistry {
    entries: Vec<Arc<OpRegEntry>>,
    map: HashMap<String, Arc<OpRegEntry>>,
}

impl AttrRegistry {
    pub fn register_or_get(&mut self, name: &str) -> Arc<OpRegEntry> {
        if let Some(entry) = self.map.get(name) {
            return entry.clone();
        }
        let mut op_node = OpNode::new(name);
        op_node.set_registry_idx(self.entries.len());
        let op_node = Arc::new(op_node);
        let entry = Arc::new(OpRegEntry {
            name: name.to_string(),
            op_node: op_node.clone(),
        });
        self.entries.push(entry.clone());
        self.map.insert(name.to_string(), entry.clone());
        entry
    }
}

impl<F> Registry<F> {
    pub fn new<T: Into<String>>(name: T, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

lazy_static! {
    /// This is an example for using doc comment attributes
    static ref REGISTRY_0: Registry<fn(Expr, Expr) -> Call> = Registry::new("0", |lhs, rhs| {
        Call::make("0", &[lhs, rhs])
    });
}
