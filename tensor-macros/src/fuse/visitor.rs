use std::collections::HashMap;

use quote::ToTokens;
use syn::{ spanned::Spanned, visit::* };

use super::{node::{ Binary, Node, Unary }, ssa::SSAContext};

pub(crate) struct Visitor<'ast> {
    pub(crate) visitor: _Visitor<'ast>,
}

impl<'ast> Visitor<'ast> {
    pub(crate) fn new() -> Self {
        Self { visitor: _Visitor::new() }
    }
    pub(crate) fn remove_unused(&mut self) {
        self.visitor.remove_unused();
    }
}

impl<'ast> Visit<'ast> for Visitor<'ast> {
    fn visit_block(&mut self, i: &'ast syn::Block) {
        visit_block(&mut self.visitor, i);
    }
}

pub(crate) struct _Visitor<'ast> {
    pub(crate) nodes: Vec<Node<'ast>>,
    pub(crate) intermidiate_var_cnt: usize,
    pub(crate) current_var: proc_macro2::Ident,
    pub(crate) current_assignment: Option<proc_macro2::Ident>,
    pub(crate) variables: HashMap<syn::Ident, bool>,
    pub(crate) next_visitor: Option<Box<_Visitor<'ast>>>,
    pub(crate) ssa_ctx: SSAContext,
}

impl<'ast> _Visitor<'ast> {
    pub(crate) fn new() -> Self {
        use proc_macro2::Span;
        Self {
            nodes: vec![],
            intermidiate_var_cnt: 0,
            current_var: proc_macro2::Ident::new("__out0", Span::call_site()),
            current_assignment: None,
            variables: HashMap::new(),
            next_visitor: None,
            ssa_ctx: SSAContext::new(),
        }
    }
    pub(crate) fn declare_variable(&mut self, ident: syn::Ident) {
        self.variables.insert(ident, false);
    }
    pub(crate) fn mark_used(&mut self, name: &syn::Ident) {
        if let Some(usage) = self.variables.get_mut(name) {
            *usage = true;
        }
    }
    pub(crate) fn mark_path_used(&mut self, path: &syn::Path) {
        if let Some(ident) = path.get_ident() {
            self.mark_used(ident);
        }
    }

    pub(crate) fn mark_expr_used(&mut self, expr: &syn::Expr) {
        match expr {
            syn::Expr::Path(path) => {
                self.mark_path_used(&path.path);
            }
            syn::Expr::Reference(reference) => {
                self.mark_expr_used(&reference.expr);
            }
            _ => {}
        }
    }
    pub(crate) fn get_unused_vars(&self) -> Vec<syn::Ident> {
        let mut unused = Vec::new();
        for (ident, usage) in &self.variables {
            if !*usage {
                unused.push(ident.clone());
            }
        }
        unused
    }

    pub(crate) fn remove_unused(&mut self) {
        let unused = self.get_unused_vars();
        self.nodes.retain(|node| {
            match node {
                Node::Unary(unary) => !unused.contains(&unary.output),
                Node::Binary(binary) => !unused.contains(&binary.output),
                Node::Input(_) => true,
            }
        });
        if let Some(next_visitor) = &mut self.next_visitor {
            next_visitor.remove_unused();
        }
    }
}

impl<'ast> Visit<'ast> for _Visitor<'ast> {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            if let syn::Pat::Ident(syn::PatIdent { ident, .. }) = &local.pat {
                let ssa_name = self.ssa_ctx.fresh_name(&ident.to_string());
                self.current_assignment = Some(proc_macro2::Ident::new(&ssa_name, ident.span()));
                self.declare_variable(ident.clone());
            }
            self.visit_expr(&init.expr);
            self.current_assignment = None;
        } else {
            unimplemented!("only support assignment for now");
        }
    }
    fn visit_block(&mut self, i: &'ast syn::Block) {
        let mut next_visitor = _Visitor::new();
        visit_block(&mut next_visitor, i);
        self.next_visitor = Some(Box::new(next_visitor));
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let mut outs: Vec<syn::Ident> = vec![];
        let current_assignment = self.current_assignment.clone();
        self.mark_expr_used(&node.receiver);
        self.visit_expr(&node.receiver);
        for arg in &node.args {
            self.mark_expr_used(arg);
            self.visit_expr(arg);
            let mut same = false;
            if let Some(out) = outs.last() {
                if out == &self.current_var {
                    same = true;
                }
            }
            if !same {
                outs.push(self.current_var.clone());
            }
        }

        let out = if let Some(current_assignment) = current_assignment {
            self.current_assignment = None;
            current_assignment
        } else {
            let out = proc_macro2::Ident::new(&format!("__out{}", self.intermidiate_var_cnt), node.span());
            self.current_var = out.clone();
            self.intermidiate_var_cnt += 1;
            out
        };
        let operand = self.ssa_ctx.current_name(&node.receiver.to_token_stream().to_string()).unwrap();
        let method = match node.method.to_string().as_str() {
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
            | "relu" => {
                Node::Unary(Unary {
                    method: &node.method,
                    operand: proc_macro2::Ident::new(&operand, node.span()),
                    output: out.clone(),
                })
            }
            _ => todo!(),
        };
        self.nodes.push(method);
        // println!("{:#?}", self.nodes);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.visit_expr(&*node.func);
        for el in syn::punctuated::Punctuated::pairs(&node.args) {
            let it = el.value();
            self.mark_expr_used(it);
            self.visit_expr(it);
        }
    }

    fn visit_expr_tuple(&mut self, tuple: &'ast syn::ExprTuple) {
        for it in &tuple.attrs {
            self.visit_attribute(it);
        }
        for el in syn::punctuated::Punctuated::pairs(&tuple.elems) {
            let it = el.value();
            self.mark_expr_used(it);
            self.visit_expr(it);
        }
    }

    fn visit_ident(&mut self, i: &'ast proc_macro2::Ident) {
        self.current_var = i.clone();
    }

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if i.path.get_ident().is_some() {
            self.visit_ident(&i.path.segments[0].ident);
        } else {
            visit_expr_path(self, i);
        }
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        let current_assignment = self.current_assignment.clone();
        self.current_assignment = None;
        self.mark_expr_used(&i.left);
        self.visit_expr(&i.left);
        let left_var = self.current_var.clone();
        self.current_assignment = None;
        self.mark_expr_used(&i.right);
        self.visit_expr(&i.right);
        let right_var = self.current_var.clone();
        let method = match i.op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => "div",
            syn::BinOp::Rem(_) => "rem",
            syn::BinOp::And(_) => "and",
            syn::BinOp::Or(_) => "or",
            syn::BinOp::BitXor(_) => "bitxor",
            syn::BinOp::BitAnd(_) => "bitand",
            syn::BinOp::BitOr(_) => "bitor",
            syn::BinOp::Shl(_) => "shl",
            syn::BinOp::Shr(_) => "shr",
            syn::BinOp::Eq(_) => "eq",
            syn::BinOp::Lt(_) => "lt",
            syn::BinOp::Le(_) => "le",
            syn::BinOp::Ne(_) => "ne",
            syn::BinOp::Ge(_) => "ge",
            syn::BinOp::Gt(_) => "gt",
            syn::BinOp::AddAssign(_) => "add_assign",
            syn::BinOp::SubAssign(_) => "sub_assign",
            syn::BinOp::MulAssign(_) => "mul_assign",
            syn::BinOp::DivAssign(_) => "div_assign",
            syn::BinOp::RemAssign(_) => "rem_assign",
            syn::BinOp::BitXorAssign(_) => "bitxor_assign",
            syn::BinOp::BitAndAssign(_) => "bitand_assign",
            syn::BinOp::BitOrAssign(_) => "bitor_assign",
            syn::BinOp::ShlAssign(_) => "shl_assign",
            syn::BinOp::ShrAssign(_) => "shr_assign",
            _ => todo!(),
        };
        let out = if let Some(current_assignment) = current_assignment {
            self.current_assignment = None;
            current_assignment
        } else {
            let out = proc_macro2::Ident::new(&format!("__out{}", self.intermidiate_var_cnt), i.span());
            self.current_var = out.clone();
            self.intermidiate_var_cnt += 1;
            out
        };
        self.nodes.push(
            Node::Binary(Binary {
                method: proc_macro2::Ident::new(method, i.span()),
                left: left_var,
                right: right_var,
                output: out.clone(),
            })
        );
        self.current_var = out;
        self.intermidiate_var_cnt += 1;
    }
}
