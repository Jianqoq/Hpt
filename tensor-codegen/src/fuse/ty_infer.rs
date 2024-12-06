use std::collections::{ HashMap, HashSet };
use syn::spanned::Spanned;
use petgraph::graph::NodeIndex;
use quote::ToTokens;
use syn::visit::Visit;

use super::{ cfg::CFG, expr_ty, operator_lists::UNARY_OPERATORS };

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

#[derive(Clone)]
pub(crate) struct TyInfer {
    pub(crate) table: HashMap<syn::Ident, Type>,
    pub(crate) visited: HashSet<NodeIndex>,
}

impl std::fmt::Debug for TyInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TyInfer")
            .field(
                "table",
                &self.table
                    .iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect::<HashMap<_, _>>()
            )
            .finish()
    }
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
                    *self.table.get(&ident).unwrap_or(&Type::Unknown)
                } else {
                    Type::Unknown
                }
            }
            syn::Expr::Lit(_) => Type::Scalar,
            syn::Expr::Try(try_expr) => self.type_of(&try_expr.expr),
            syn::Expr::Call(_) => { Type::Unknown }
            syn::Expr::MethodCall(method_call) => {
                let receiver_type = self.type_of(&method_call.receiver);
                if receiver_type == Type::Tensor {
                    let func_name = method_call.method.to_token_stream().to_string();
                    if UNARY_OPERATORS.contains(&func_name.as_str()) {
                        return Type::Tensor;
                    }
                    Type::Unknown
                } else {
                    Type::Unknown
                }
            }
            syn::Expr::Paren(paren) => self.type_of(&paren.expr),
            syn::Expr::Tuple(_) => Type::Unknown,
            _ => unimplemented!("ty_infer::type_of::{:#?}", expr_ty::ExprType::from(expr)),
        }
    }

    pub(crate) fn infer(&mut self, cfg: &CFG) -> anyhow::Result<()> {
        self._infer(cfg, cfg.entry)?;
        Ok(())
    }

    fn _infer(&mut self, cfg: &CFG, node: NodeIndex) -> anyhow::Result<()> {
        if let Some(block) = cfg.graph.node_weight(node) {
            for phi_function in &block.phi_functions {
                let first_arg = &phi_function.args[0];
                self.table.insert(phi_function.name.clone(), match self.table.get(first_arg) {
                    Some(ty) => ty.clone(),
                    None => {
                        return Err(
                            syn::Error
                                ::new(
                                    first_arg.span(),
                                    &format!(
                                        "Internal: can't find type for variable {}",
                                        first_arg.to_string()
                                    )
                                )
                                .into()
                        );
                    }
                });
            }
            for stmt in &block.statements {
                self.visit_stmt(&stmt.stmt);
            }
            for succ in cfg.graph.neighbors_directed(node, petgraph::Direction::Outgoing) {
                if self.visited.contains(&succ) {
                    continue;
                }
                self.visited.insert(succ);
                self._infer(cfg, succ)?;
            }
        }
        Ok(())
    }
}

fn handle_pat_ty(lhs: &syn::Pat, rhs: &syn::Type, table: &mut HashMap<syn::Ident, Type>) {
    match (lhs, rhs) {
        (syn::Pat::Ident(pat_ident), syn::Type::Path(type_path)) => {
            let path = type_path.path.segments
                .last()
                .expect("handle_pat_ty::path::no_segment::last_segment::110")
                .ident.to_string();
            if path.contains("Tensor") || path.contains("_Tensor") {
                table.insert(pat_ident.ident.clone(), Type::Tensor);
            } else {
                table.insert(pat_ident.ident.clone(), Type::Unknown);
            }
        }
        (syn::Pat::Ident(pat_ident), _) => {
            table.insert(pat_ident.ident.clone(), Type::Unknown);
        }
        (syn::Pat::Path(expr_path), syn::Type::Path(type_path)) => {
            let path = type_path.path.segments
                .last()
                .expect("handle_pat_ty::path::no_segment::last_segment::120")
                .ident.to_string();
            if path.contains("Tensor") || path.contains("_Tensor") {
                table.insert(
                    expr_path.path.segments
                        .last()
                        .expect("handle_pat_ty::path::no_segment::last_segment::123")
                        .ident.clone(),
                    Type::Tensor
                );
            } else {
                table.insert(
                    expr_path.path.segments
                        .last()
                        .expect("handle_pat_ty::path::no_segment::last_segment::125")
                        .ident.clone(),
                    Type::Unknown
                );
            }
        }
        (syn::Pat::Path(expr_path), _) => {
            if let Some(ident) = expr_path.path.get_ident() {
                table.insert(ident.clone(), Type::Unknown);
            } else {
                panic!("handle_pat_ty::path::no_ident");
            }
        }
        (syn::Pat::Reference(pat_reference), _) => {
            handle_pat_ty(pat_reference.pat.as_ref(), rhs, table);
        }
        (syn::Pat::Tuple(pat_tuple), syn::Type::Tuple(type_tuple)) => {
            for (lhs, rhs) in pat_tuple.elems.iter().zip(type_tuple.elems.iter()) {
                handle_pat_ty(lhs, rhs, table);
            }
        }
        (syn::Pat::Wild(_), _) => {
            table.insert(syn::Ident::new("_", lhs.span()), Type::Unknown);
        }
        _ =>
            unreachable!(
                "handle_pat_ty::{:#?}: {:#?}",
                lhs.to_token_stream(),
                rhs.to_token_stream()
            ),
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
                        let ident = pat_ident.ident.clone();
                        if tys.iter().any(|t| ty == *t) {
                            self.table.insert(ident, Type::Scalar);
                        } else if
                            ident.to_string().contains("Tensor") ||
                            ident.to_string().contains("_Tensor")
                        {
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
                let ident = pat_ident.ident.clone();
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
                self.visit_pat_type(ty);
            }
            syn::Pat::Verbatim(_) => unimplemented!("ty_infer::visit_local::Verbatim"),
            syn::Pat::Wild(wild) => {
                self.table.insert(
                    syn::Ident::new("_", wild.underscore_token.span()),
                    Type::Unknown
                );
            }
            _ => unimplemented!("ty_infer::visit_local::{}", i.pat.to_token_stream().to_string()),
        }
    }

    fn visit_pat_type(&mut self, i: &'ast syn::PatType) {
        handle_pat_ty(i.pat.as_ref(), i.ty.as_ref(), &mut self.table);
    }

    fn visit_expr_assign(&mut self, _: &'ast syn::ExprAssign) {}
}
