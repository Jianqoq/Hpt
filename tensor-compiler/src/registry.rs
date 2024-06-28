use std::sync::Mutex;

use hashbrown::HashMap;
use lazy_static::lazy_static;

use crate::{ halide::{ exprs::Call, prime_expr::PrimeExpr, variable::Variable }, hlir::exprs::OpNode, op::OpType };

pub(crate) fn add_binop(registory: &mut AttrRegistry, name: &str) {
    let op_node = registory.register_or_get(name);
    op_node.set_num_inputs(2);
    op_node.add_argument("lhs", "Tensor", "");
    op_node.add_argument("rhs", "Tensor", "");
    op_node.set_op_type(OpType::OneToMany);
}

pub(crate) fn add_unop(registry: &mut AttrRegistry, name: &str) {
    let op_node = registry.register_or_get(name);
    op_node.set_num_inputs(1);
    op_node.add_argument("lhs", "Tensor", "");
    op_node.set_op_type(OpType::OneToOne);
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub enum Closures {
    Common(fn(Vec<PrimeExpr>) -> Call),
    Init(fn(Variable, PrimeExpr) -> Call),
}

pub enum ClosuresType {
    Common,
    Unop,
}

impl Closures {
    pub fn get_binop_expr<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(&self, lhs: A, rhs: B) -> Call {
        match self {
            Closures::Common(f) => f(vec![lhs.into(), rhs.into()]),
            _ => panic!("not binop"),
        }
    }
    pub fn call_common(&self, vec: Vec<PrimeExpr>) -> Call {
        match self {
            Closures::Common(f) => f(vec),
            _ => panic!("not common"),
        }
    }
}

pub struct Manager {
    map: HashMap<String, Closures>,
}

impl Manager {
    pub fn get(&self, name: &str) -> Option<&Closures> {
        self.map.get(name)
    }
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
    pub static ref REGISTRY: Mutex<AttrRegistry> = {
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
        add_binop(&mut ret, "sum");
        add_unop(&mut ret, "not");
        add_unop(&mut ret, "neg");
        add_unop(&mut ret, "abs");
        add_unop(&mut ret, "sqrt");
        add_unop(&mut ret, "exp");
        add_unop(&mut ret, "log");
        add_unop(&mut ret, "log2");
        add_unop(&mut ret, "log10");
        add_unop(&mut ret, "sin");
        add_unop(&mut ret, "cos");
        add_unop(&mut ret, "tan");
        add_unop(&mut ret, "asin");
        add_unop(&mut ret, "acos");
        add_unop(&mut ret, "atan");
        add_unop(&mut ret, "sinh");
        add_unop(&mut ret, "cosh");
        add_unop(&mut ret, "tanh");
        add_unop(&mut ret, "asinh");
        add_unop(&mut ret, "acosh");
        add_unop(&mut ret, "atanh");

        ret.into()
    };
}

macro_rules! binop {
    ($ret:ident, $op_name:ident) => {
        $ret.map.insert(stringify!($op_name).to_string(), Closures::Common(|vec| {
            let locked = REGISTRY.lock().unwrap();
            let op = locked.map.get(stringify!($op_name)).expect(&format!("{} not found", stringify!($op_name)));
            Call::make(op.name(), &[vec[0].clone(), vec[1].clone()])
        }));
    };
}

macro_rules! unop {
    ($ret:ident, $op_name:ident) => {
        $ret.map.insert(stringify!($op_name).to_string(), Closures::Common(|vec| {
            let locked = REGISTRY.lock().unwrap();
            let op = locked.map.get(stringify!($op_name)).unwrap();
            Call::make(op.name(), &[vec[0].clone()])
        }));
    };
}

lazy_static! {
    pub static ref MANAGER: Mutex<Manager> = {
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
        binop!(ret, sum);
        binop!(ret, power);
        binop!(ret, eq);
        binop!(ret, ne);
        binop!(ret, lt);
        binop!(ret, le);
        binop!(ret, gt);
        binop!(ret, ge);
        unop!(ret, not);
        unop!(ret, neg);
        unop!(ret, abs);
        unop!(ret, sqrt);
        unop!(ret, exp);
        unop!(ret, log);
        unop!(ret, log2);
        unop!(ret, log10);
        unop!(ret, sin);
        unop!(ret, cos);
        unop!(ret, tan);
        unop!(ret, asin);
        unop!(ret, acos);
        unop!(ret, atan);
        unop!(ret, sinh);
        unop!(ret, cosh);
        unop!(ret, tanh);
        unop!(ret, asinh);
        unop!(ret, acosh);
        unop!(ret, atanh);
        ret.into()
    };
}
