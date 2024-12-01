use std::collections::{ HashMap, HashSet };
use proc_macro2::TokenTree;
use quote::ToTokens;
use syn::visit::Visit;

use super::{ expr_ty, ty_infer::Type, use_define_visitor::UseDefineVisitor };

pub(crate) struct ExprCallUseVisitor<'ast> {
    table: &'ast HashMap<syn::Ident, Type>,
    pub(crate) used_vars: HashSet<syn::Ident>,
}

impl<'ast> ExprCallUseVisitor<'ast> {
    pub(crate) fn new(table: &'ast HashMap<syn::Ident, Type>) -> Self {
        Self {
            table,
            used_vars: HashSet::new(),
        }
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
                    if ["shape", "strides"].contains(&func_name.as_str()) {
                        return Type::Unknown;
                    }
                    Type::Tensor
                } else {
                    Type::Unknown
                }
            }
            _ =>
                unimplemented!("ExprCallUseVisitor::type_of::{:#?}", expr_ty::ExprType::from(expr)),
        }
    }
}

impl<'ast> Visit<'ast> for ExprCallUseVisitor<'ast> {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let mut var_collector = UseDefineVisitor::new();
        var_collector.visit_expr_call(call);
        self.used_vars.extend(var_collector.used_vars);
    }
    fn visit_expr_method_call(&mut self, method_call: &'ast syn::ExprMethodCall) {
        let methods = [
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "relu",
            "selu",
        ];
        if
            self.type_of(&method_call.receiver) == Type::Tensor &&
            methods.contains(&method_call.method.to_string().as_str())
        {
            return;
        }
        let mut var_collector = UseDefineVisitor::new();
        var_collector.visit_expr_method_call(method_call);
        self.used_vars.extend(var_collector.used_vars);
    }

    fn visit_stmt_macro(&mut self, macro_stmt: &'ast syn::StmtMacro) {
        let tokens = macro_stmt.mac.tokens.clone();
        for arg in tokens.into_iter() {
            if let TokenTree::Ident(ident) = arg {
                self.used_vars.insert(ident);
            }
        }
    }
}
