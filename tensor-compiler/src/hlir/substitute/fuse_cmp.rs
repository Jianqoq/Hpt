#![allow(dead_code)]
use hashbrown::HashMap;
use tensor_common::shape::Shape;

use crate::{
    halide::variable::Variable,
    hlir::{ expr::Expr, exprs::Tensor, traits::{ HlirMutateVisitor, MutatorGetSet } },
};

pub struct FuseComputeNode {
    map: HashMap<Variable, Tensor>,
    id: usize,
    expr: Expr,
}

impl FuseComputeNode {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            id: 0,
            expr: Expr::None,
        }
    }
}

impl MutatorGetSet for FuseComputeNode {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }
}

macro_rules! impl_binop_call {
    ($var:ident, $method:expr, $args:ident, $id:expr, $map:expr, $err_msg:expr) => {
        assert!($args.len() == 2, $err_msg);
        let lhs = $args[0].to_tensor().unwrap().clone();
        let rhs = $args[1].to_tensor().unwrap().clone();
        let tensor = Tensor::make_binop($method, lhs, rhs, $id);
        $map.insert($var.into(), tensor);
    };
}

macro_rules! impl_reduce_call {
    ($var:ident, $method:expr, $args:ident, $id:expr, $map:expr, $err_msg:expr) => {
        assert!($args.len() == 3, $err_msg);
        let to_reduce = $args[0].to_tensor().unwrap().clone();
        let axes = $args[1].to_tuple().unwrap().clone();
        let initial = $args[2].clone();
        let tensor = Tensor::make_reduce($method, to_reduce, axes, initial.to_primexpr().expect("expect prim"), $id);
        $map.insert($var.into(), tensor);
    };
}

macro_rules! impl_unop_call {
    ($var:ident, $method:expr, $args:ident, $id:expr, $map:expr, $err_msg:expr) => {
        assert!($args.len() == 1, $err_msg);
        let tensor = $args[0].to_tensor().unwrap().clone();
        let new_tensor = Tensor::make_unop($method, tensor, $id);
        $map.insert($var.into(), new_tensor);
    };
}

impl HlirMutateVisitor for FuseComputeNode {
    fn visit_let(&mut self, let_: &crate::hlir::exprs::Let) {
        let var = let_.var();
        let value = let_.value();
        let body = let_.body();
        match value {
            Expr::Value(val) => {
                let mut tensor = Tensor::make(Shape::from([1]), val.dtype(), self.id);
                tensor.set_value(val);
                self.map.insert(var.into(), tensor);
            }
            Expr::Str(_) => todo!(),
            Expr::Variable(var) => {
                if let Some(tensor) = self.map.get(var) {
                    self.map.insert(var.into(), tensor.clone());
                } else {
                    panic!("variable {} not found in map", var);
                }
            }
            Expr::Tuple(_) => todo!(),
            Expr::Type(_) => unreachable!(),
            Expr::TensorType(_) => unreachable!(),
            Expr::OpNode(_) => unreachable!(),
            Expr::Tensor(tensor) => {
                self.map.insert(var.into(), tensor.clone());
            }
            Expr::Cast(cast) => {
                let val = cast.value();
                self.map.insert(
                    var.into(),
                    Tensor::make_const(cast.dtype(), val.casted_value(cast.dtype()), self.id)
                );
            }
            Expr::Not(_) => todo!(),
            Expr::Call(call) => {
                let args = call.args();
                let mut tensor_args = Vec::<Expr>::new();
                for arg in args {
                    match arg {
                        Expr::Value(_) => {
                            tensor_args.push(arg.clone());
                        }
                        Expr::Str(_) => {
                            tensor_args.push(arg.clone());
                        }
                        Expr::Variable(var) => {
                            if let Some(tensor) = self.map.get(var) {
                                tensor_args.push(tensor.into());
                            } else {
                                panic!("call argument {} not found in map", arg);
                            }
                        }
                        Expr::Tuple(_) => {
                            tensor_args.push(arg.clone());
                        }
                        Expr::Tensor(_) => {
                            tensor_args.push(arg.clone());
                        }
                        _ => unreachable!(),
                    }
                }
                let call_method = call.name();
                match call_method {
                    Expr::Str(_) => todo!(),
                    Expr::Variable(call_method) => {
                        match call_method.name() {
                            "add" => { impl_binop_call!(var, "add", tensor_args, self.id, self.map, "add requires 2 arguments"); } // prettier-ignore
                            "sub" => { impl_binop_call!(var, "sub", tensor_args, self.id, self.map, "sub requires 2 arguments"); } // prettier-ignore
                            "mul" => { impl_binop_call!(var, "mul", tensor_args, self.id, self.map, "mul requires 2 arguments"); } // prettier-ignore
                            "div" => { impl_binop_call!(var, "div", tensor_args, self.id, self.map, "div requires 2 arguments"); } // prettier-ignore
                            "mod" => { impl_binop_call!(var, "mod", tensor_args, self.id, self.map, "mod requires 2 arguments"); } // prettier-ignore
                            "eq" => { impl_binop_call!(var, "eq", tensor_args, self.id, self.map, "eq requires 2 arguments"); } // prettier-ignore
                            "ne" => { impl_binop_call!(var, "ne", tensor_args, self.id, self.map, "ne requires 2 arguments"); } // prettier-ignore
                            "lt" => { impl_binop_call!(var, "lt", tensor_args, self.id, self.map, "lt requires 2 arguments"); } // prettier-ignore
                            "le" => { impl_binop_call!(var, "le", tensor_args, self.id, self.map, "le requires 2 arguments"); } // prettier-ignore
                            "gt" => { impl_binop_call!(var, "gt", tensor_args, self.id, self.map, "gt requires 2 arguments"); } // prettier-ignore
                            "ge" => { impl_binop_call!(var, "ge", tensor_args, self.id, self.map, "ge requires 2 arguments"); } // prettier-ignore
                            "and" => { impl_binop_call!(var, "and", tensor_args, self.id, self.map, "and requires 2 arguments"); } // prettier-ignore
                            "or" => { impl_binop_call!(var, "or", tensor_args, self.id, self.map, "or requires 2 arguments"); } // prettier-ignore
                            "xor" => { impl_binop_call!(var, "xor", tensor_args, self.id, self.map, "xor requires 2 arguments"); } // prettier-ignore
                            "sum" => { impl_reduce_call!(var, "sum", tensor_args, self.id, self.map, "sum requires 3 argument, 1: tensor to reduce, 2: axes, 3: init_val"); } // prettier-ignore
                            "max" => { impl_reduce_call!(var, "max", tensor_args, self.id, self.map, "max requires 3 argument, 1: tensor to reduce, 2: axes, 3: init_val"); } // prettier-ignore
                            "min" => { impl_reduce_call!(var, "min", tensor_args, self.id, self.map, "min requires 3 argument, 1: tensor to reduce, 2: axes, 3: init_val"); } // prettier-ignore
                            "reshape" => {
                                assert!(tensor_args.len() == 2, "reshape requires 2 argument, 1: tensor to reshape, 2: new shape"); // prettier-ignore
                                let tensor = tensor_args[0].to_tensor().unwrap().clone();
                                let tuple = tensor_args[1].to_tuple().unwrap().to_shape();
                                if let Some(shape) = tuple {
                                    let mut new_tensor = tensor.clone();
                                    new_tensor.reshape(&shape);
                                    self.map.insert(var.into(), new_tensor);
                                } else {
                                    panic!("tuple cannot be converted to shape");
                                }
                            }
                            "exp" => { impl_unop_call!(var, "exp", tensor_args, self.id, self.map, "exp requires 1 argument, 1: tensor to apply exp"); } // prettier-ignore
                            "log" => { impl_unop_call!(var, "log", tensor_args, self.id, self.map, "log requires 1 argument, 1: tensor to apply log"); } // prettier-ignore
                            "abs" => { impl_unop_call!(var, "abs", tensor_args, self.id, self.map, "abs requires 1 argument, 1: tensor to apply abs"); } // prettier-ignore
                            "sqrt" => { impl_unop_call!(var, "sqrt", tensor_args, self.id, self.map, "sqrt requires 1 argument, 1: tensor to apply sqrt"); } // prettier-ignore
                            "sin" => { impl_unop_call!(var, "sin", tensor_args, self.id, self.map, "sin requires 1 argument, 1: tensor to apply sin"); } // prettier-ignore
                            "cos" => { impl_unop_call!(var, "cos", tensor_args, self.id, self.map, "cos requires 1 argument, 1: tensor to apply cos"); } // prettier-ignore
                            "tan" => { impl_unop_call!(var, "tan", tensor_args, self.id, self.map, "tan requires 1 argument, 1: tensor to apply tan"); } // prettier-ignore
                            "asin" => { impl_unop_call!(var, "asin", tensor_args, self.id, self.map, "asin requires 1 argument, 1: tensor to apply asin"); } // prettier-ignore
                            "acos" => { impl_unop_call!(var, "acos", tensor_args, self.id, self.map, "acos requires 1 argument, 1: tensor to apply acos"); } // prettier-ignore
                            "atan" => { impl_unop_call!(var, "atan", tensor_args, self.id, self.map, "atan requires 1 argument, 1: tensor to apply atan"); } // prettier-ignore
                            "sinh" => { impl_unop_call!(var, "sinh", tensor_args, self.id, self.map, "sinh requires 1 argument, 1: tensor to apply sinh"); } // prettier-ignore
                            "cosh" => { impl_unop_call!(var, "cosh", tensor_args, self.id, self.map, "cosh requires 1 argument, 1: tensor to apply cosh"); } // prettier-ignore
                            "tanh" => { impl_unop_call!(var, "tanh", tensor_args, self.id, self.map, "tanh requires 1 argument, 1: tensor to apply tanh"); } // prettier-ignore
                            "asinh" => { impl_unop_call!(var, "asinh", tensor_args, self.id, self.map, "asinh requires 1 argument, 1: tensor to apply asinh"); } // prettier-ignore
                            "acosh" => { impl_unop_call!(var, "acosh", tensor_args, self.id, self.map, "acosh requires 1 argument, 1: tensor to apply acosh"); } // prettier-ignore
                            "atanh" => { impl_unop_call!(var, "atanh", tensor_args, self.id, self.map, "atanh requires 1 argument, 1: tensor to apply atanh"); } // prettier-ignore
                            _ => unreachable!("call method {} not found", call_method.name()),
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!("unexpected expr {:?}", value),
        }
        self.id += 1;
        self.visit_expr(body);
    }

    fn visit_function(&mut self, func: &crate::hlir::exprs::Function) {
        let args = func.args();
        let body = func.body();
        for arg in args {
            if !self.map.contains_key(arg) {
                panic!("function argument {} not found in map", arg);
            }
        }
        self.visit_expr(body);
    }
}
