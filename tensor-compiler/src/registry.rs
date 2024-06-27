use std::sync::Mutex;

use hashbrown::HashMap;
use lazy_static::lazy_static;

use crate::{ hlir::{ expr::Expr, exprs::{ Call, OpNode } }, op::OpType };

pub(crate) fn add_binop(registory: &mut AttrRegistry, name: &str) {
    let op_node = registory.register_or_get(name);
    op_node.set_num_inputs(2);
    op_node.add_argument("lhs", "Tensor", "");
    op_node.add_argument("rhs", "Tensor", "");
    op_node.set_op_type(OpType::OneToMany);
}

pub struct Registry<F> {
    name: String,
    f: F,
}

pub struct AttrRegistry {
    map: HashMap<String, Box<OpNode>>,
}

impl AttrRegistry {
    pub fn register_or_get(&mut self, name: &str) -> &mut OpNode {
        self.map.entry(name.to_string()).or_insert_with(|| Box::new(OpNode::new(name)))
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
    static ref REGISTRY: Mutex<AttrRegistry> = {
        let mut ret = AttrRegistry {
            map: HashMap::new(),
        };
        add_binop(&mut ret, "add");
        add_binop(&mut ret, "sub");
        add_binop(&mut ret, "mul");
        add_binop(&mut ret, "div");
        add_binop(&mut ret, "mod");
        add_binop(&mut ret, "left_shift");
        add_binop(&mut ret, "right_shift");
        add_binop(&mut ret, "and");
        add_binop(&mut ret, "or");
        add_binop(&mut ret, "xor");
        add_binop(&mut ret, "max");
        add_binop(&mut ret, "min");
        add_binop(&mut ret, "power");
        add_binop(&mut ret, "eq");
        add_binop(&mut ret, "ne");
        add_binop(&mut ret, "lt");
        add_binop(&mut ret, "le");
        add_binop(&mut ret, "gt");
        add_binop(&mut ret, "ge");
        ret.into()
    };
}

macro_rules! binop {
    ($var_name:ident, $op_name:ident) => {
        lazy_static! {
            static ref $var_name: Registry<fn(Expr, Expr) -> Call> = Registry::new(stringify!($op_name), |lhs, rhs| {
                Call::make(stringify!($op_name), &[lhs, rhs])
            });
        }
    };
}

binop!(REGISTRY_0, add);
binop!(REGISTRY_1, sub);
binop!(REGISTRY_2, mul);
binop!(REGISTRY_3, div);
binop!(REGISTRY_4, mod);
binop!(REGISTRY_5, left_shift);
binop!(REGISTRY_6, right_shift);
binop!(REGISTRY_7, and);
binop!(REGISTRY_8, or);
binop!(REGISTRY_9, xor);
binop!(REGISTRY_10, max);
binop!(REGISTRY_11, min);
binop!(REGISTRY_12, power);
binop!(REGISTRY_13, eq);
binop!(REGISTRY_14, ne);
binop!(REGISTRY_15, lt);
binop!(REGISTRY_16, le);
binop!(REGISTRY_17, gt);
binop!(REGISTRY_18, ge);
