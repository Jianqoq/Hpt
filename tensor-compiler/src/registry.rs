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

pub enum Closures {
    Binop(fn(Expr, Expr) -> Call),
}

pub struct Manager {
    map: HashMap<String, Closures>,
}

pub struct AttrRegistry {
    map: HashMap<String, Box<OpNode>>,
}

impl AttrRegistry {
    pub fn register_or_get(&mut self, name: &str) -> &mut OpNode {
        self.map.entry(name.to_string()).or_insert_with(|| Box::new(OpNode::new(name)))
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
    ($ret:ident, $op_name:ident) => {
        $ret.map.insert(stringify!($op_name).to_string(), Closures::Binop(|lhs, rhs| {
            Call::make(stringify!($op_name), &[lhs, rhs])
        }));
    };
}

lazy_static! {
    static ref MANAGER: Mutex<Manager> = {
        let mut ret = Manager {
            map: HashMap::new(),
        };
        binop!(ret, add);
        binop!(ret, sub);
        binop!(ret, mul);
        binop!(ret, div);
        binop!(ret, mod);
        binop!(ret, left_shift);
        binop!(ret, right_shift);
        binop!(ret, and);
        binop!(ret, or);
        binop!(ret, xor);
        binop!(ret, max);
        binop!(ret, min);
        binop!(ret, power);
        binop!(ret, eq);
        binop!(ret, ne);
        binop!(ret, lt);
        binop!(ret, le);
        binop!(ret, gt);
        binop!(ret, ge);
        ret.into()
    };
}

pub trait Callable {
    fn call(&self);
}
