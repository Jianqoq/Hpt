use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;
use quote::ToTokens;
use syn::visit::Visit;

use super::{ cfg::CFG, expr_ty };

#[derive(Debug, Clone, PartialEq, Copy)]
pub(crate) enum Type {
    Scalar,
    Tensor,
    Unknown,
}

impl Type {
    pub(crate) fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TyInfer {
    pub(crate) table: HashMap<String, Type>,
    pub(crate) visited: HashSet<NodeIndex>,
}

impl TyInfer {
    pub(crate) fn new() -> Self {
        Self { table: HashMap::new(), visited: HashSet::new() }
    }

    pub(crate) fn type_of(&self, expr: &syn::Expr) -> Type {
        match expr {
            syn::Expr::Binary(node) => {
                let left = self.type_of(&node.left);
                let right = self.type_of(&node.right);
                match (left, right) {
                    (Type::Scalar, Type::Scalar) => Type::Scalar,
                    (Type::Scalar, Type::Tensor) => Type::Tensor,
                    (Type::Scalar, Type::Unknown) => Type::Unknown,
                    (Type::Tensor, Type::Scalar) => Type::Tensor,
                    (Type::Tensor, Type::Tensor) => Type::Tensor,
                    (Type::Tensor, Type::Unknown) => Type::Tensor,
                    (Type::Unknown, Type::Scalar) => Type::Unknown,
                    (Type::Unknown, Type::Tensor) => Type::Tensor,
                    (Type::Unknown, Type::Unknown) => Type::Unknown,
                }
            }
            syn::Expr::Reference(reference) => { self.type_of(&reference.expr) }
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    *self.table.get(&ident.to_string()).unwrap_or(&Type::Unknown)
                } else {
                    Type::Unknown
                }
            }
            syn::Expr::Lit(_) => Type::Scalar,
            syn::Expr::Try(try_expr) => self.type_of(&try_expr.expr),
            syn::Expr::Call(call) => {
                let func_name = call.func.to_token_stream().to_string();
                if func_name == "__phi" {
                    let arg_type = self.type_of(&call.args[0]);
                    for arg in call.args.iter() {
                        let arg_type = self.type_of(&arg);
                        if arg_type != arg_type {
                            panic!("phi function's arguments must have the same type");
                        }
                    }
                    arg_type
                } else {
                    Type::Unknown
                }
            }
            syn::Expr::MethodCall(method_call) => {
                let receiver_type = self.type_of(&method_call.receiver);
                if receiver_type == Type::Tensor {
                    let func_name = method_call.method.to_token_stream().to_string();
                    if ["shape", "strides"].contains(&func_name.as_str()) {
                        return Type::Unknown;
                    }
                    Type::Tensor
                } else {
                    Type::Unknown
                }
            }
            _ => unimplemented!("ty_infer::type_of::{:#?}", expr_ty::ExprType::from(expr)),
        }
    }

    pub(crate) fn infer(&mut self, cfg: &CFG) {
        self._infer(cfg, cfg.entry);
    }

    fn _infer(&mut self, cfg: &CFG, node: NodeIndex) {
        if let Some(block) = cfg.graph.node_weight(node) {
            for stmt in &block.statements {
                self.visit_stmt(&stmt.stmt);
            }
            for succ in cfg.graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                if self.visited.contains(&succ) {
                    continue;
                }
                self.visited.insert(succ);
                self._infer(cfg, succ);
            }
        }
    }
}

impl<'ast> Visit<'ast> for TyInfer {
    fn visit_signature(&mut self, i: &'ast syn::Signature) {
        for arg in i.inputs.iter() {
            if let syn::FnArg::Typed(pat_type) = arg {
                let pat = pat_type.pat.as_ref();
                let ty = pat_type.ty.to_token_stream().to_string();
                let tys = [
                    "bool",
                    "i8",
                    "i16",
                    "i32",
                    "i64",
                    "u8",
                    "u16",
                    "u32",
                    "u64",
                    "f32",
                    "f64",
                ];
                match pat {
                    syn::Pat::Ident(pat_ident) => {
                        let ident = pat_ident.ident.to_string();
                        if tys.iter().any(|t| ty == *t) {
                            self.table.insert(ident, Type::Scalar);
                        } else if ident.contains("Tensor") || ident.contains("_Tensor") {
                            self.table.insert(ident, Type::Tensor);
                        } else {
                            self.table.insert(ident, Type::Unknown);
                        }
                    }
                    syn::Pat::Lit(_) => unimplemented!("ty_infer::visit_signature::Lit"),
                    syn::Pat::Macro(_) => unimplemented!("ty_infer::visit_signature::Macro"),
                    syn::Pat::Or(_) => unimplemented!("ty_infer::visit_signature::Or"),
                    syn::Pat::Paren(_) => unimplemented!("ty_infer::visit_signature::Paren"),
                    syn::Pat::Path(_) => unimplemented!("ty_infer::visit_signature::Path"),
                    syn::Pat::Range(_) => unimplemented!("ty_infer::visit_signature::Range"),
                    syn::Pat::Reference(_) =>
                        unimplemented!("ty_infer::visit_signature::Reference"),
                    syn::Pat::Rest(_) => unimplemented!("ty_infer::visit_signature::Rest"),
                    syn::Pat::Slice(_) => unimplemented!("ty_infer::visit_signature::Slice"),
                    syn::Pat::Struct(_) => unimplemented!("ty_infer::visit_signature::Struct"),
                    syn::Pat::Tuple(_) => unimplemented!("ty_infer::visit_signature::Tuple"),
                    syn::Pat::TupleStruct(_) =>
                        unimplemented!("ty_infer::visit_signature::TupleStruct"),
                    syn::Pat::Type(_) => unimplemented!("ty_infer::visit_signature::Type"),
                    syn::Pat::Verbatim(_) => unimplemented!("ty_infer::visit_signature::Verbatim"),
                    syn::Pat::Wild(_) => unimplemented!("ty_infer::visit_signature::Wild"),
                    _ =>
                        unimplemented!(
                            "ty_infer::visit_signature::{}",
                            pat.to_token_stream().to_string()
                        ),
                }
            }
        }
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        match &i.pat {
            syn::Pat::Const(_) => unimplemented!("ty_infer::visit_local::Const"),
            syn::Pat::Ident(pat_ident) => {
                let ident = pat_ident.ident.to_string();
                if let Some(init) = &i.init {
                    self.table.insert(ident, self.type_of(&init.expr));
                } else {
                    self.table.insert(ident, Type::Unknown);
                }
            }
            syn::Pat::Lit(_) => unimplemented!("ty_infer::visit_local::Lit"),
            syn::Pat::Macro(_) => unimplemented!("ty_infer::visit_local::Macro"),
            syn::Pat::Or(_) => unimplemented!("ty_infer::visit_local::Or"),
            syn::Pat::Paren(_) => unimplemented!("ty_infer::visit_local::Paren"),
            syn::Pat::Path(_) => unimplemented!("ty_infer::visit_local::Path"),
            syn::Pat::Range(_) => unimplemented!("ty_infer::visit_local::Range"),
            syn::Pat::Reference(_) => unimplemented!("ty_infer::visit_local::Reference"),
            syn::Pat::Rest(_) => unimplemented!("ty_infer::visit_local::Rest"),
            syn::Pat::Slice(_) => unimplemented!("ty_infer::visit_local::Slice"),
            syn::Pat::Struct(_) => unimplemented!("ty_infer::visit_local::Struct"),
            syn::Pat::Tuple(_) => unimplemented!("ty_infer::visit_local::Tuple"),
            syn::Pat::TupleStruct(_) => unimplemented!("ty_infer::visit_local::TupleStruct"),
            syn::Pat::Type(ty) => {
                if let syn::Pat::Ident(pat_ident) = ty.pat.as_ref() {
                    let ident = pat_ident.ident.to_string();
                    let ty = ty.ty.to_token_stream().to_string();
                    if ty.contains("Tensor") || ty.contains("_Tensor") {
                        self.table.insert(ident, Type::Tensor);
                    } else {
                        self.table.insert(ident, Type::Unknown);
                    }
                } else {
                    unimplemented!("ty_infer::visit_local::Type::None")
                }
            }
            syn::Pat::Verbatim(_) => unimplemented!("ty_infer::visit_local::Verbatim"),
            syn::Pat::Wild(_) => {
                self.table.insert("_".to_string(), Type::Unknown);
            }
            _ => unimplemented!("ty_infer::visit_local::{}", i.pat.to_token_stream().to_string()),
        }
    }

    fn visit_expr_assign(&mut self, _: &'ast syn::ExprAssign) {}
}
