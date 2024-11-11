use std::collections::HashMap;

use quote::ToTokens;
use syn::{ spanned::Spanned, visit::* };

use super::{ dag::Var2, node::{ Binary, Node, Unary }, ssa::SSAContext };

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
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        for it in &node.attrs {
            self.visitor.visit_attribute(it);
        }
        self.visitor.visit_visibility(&node.vis);
        self.visitor.visit_signature(&node.sig);
        syn::visit::visit_block(&mut self.visitor, &*node.block);
    }
}

pub(crate) struct _Visitor<'ast> {
    pub(crate) nodes: Vec<Node<'ast>>,
    pub(crate) current_var: proc_macro2::Ident,
    pub(crate) current_assignment: Option<proc_macro2::Ident>,
    pub(crate) variables: HashMap<syn::Ident, (bool, bool)>,
    pub(crate) next_visitor: Option<Box<_Visitor<'ast>>>,
    pub(crate) ssa_ctx: SSAContext,
}

impl<'ast> _Visitor<'ast> {
    pub(crate) fn new() -> Self {
        use proc_macro2::Span;
        Self {
            nodes: vec![],
            current_var: proc_macro2::Ident::new("__out0", Span::call_site()),
            current_assignment: None,
            variables: HashMap::new(),
            next_visitor: None,
            ssa_ctx: SSAContext::new(),
        }
    }
    #[allow(unused)]
    pub(crate) fn variables(&self) -> &HashMap<syn::Ident, (bool, bool)> {
        &self.variables
    }
    pub(crate) fn declare_variable(&mut self, ident: syn::Ident, is_tensor: bool) {
        self.variables.insert(ident, (false, is_tensor));
    }
    pub(crate) fn mark_used(&mut self, name: &syn::Ident) {
        if let Some((usage, _)) = self.variables.get_mut(name) {
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
        for (ident, (usage, _)) in &self.variables {
            if !*usage {
                unused.push(ident.clone());
            }
        }
        unused
    }

    pub(crate) fn remove_unused(&mut self) {
        let unused = self.get_unused_vars();
        println!("unused: {:#?}", unused);
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

    pub(crate) fn is_tensor_expr(&self, expr: &syn::Expr) -> bool {
        match expr {
            syn::Expr::Binary(node) => {
                self.is_tensor_expr(&*node.left) || self.is_tensor_expr(&*node.right)
            }
            syn::Expr::Block(_) => unimplemented!("is_tensor_expr::block"),
            syn::Expr::Call(_) => unimplemented!("is_tensor_expr::call"),
            syn::Expr::If(_) => unimplemented!("is_tensor_expr::if"),
            syn::Expr::Macro(_) => unimplemented!("is_tensor_expr::macro"),
            syn::Expr::Match(_) => unimplemented!("is_tensor_expr::match"),
            syn::Expr::MethodCall(method_call) => {
                if self.is_tensor_expr(&method_call.receiver) {
                    match method_call.method.to_string().as_str() {
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
                        | "relu"
                        | "selu" => true,
                        _ =>
                            unimplemented!(
                                "is_tensor_expr::method_call::{}",
                                method_call.method.to_string().as_str()
                            ),
                    }
                } else {
                    false
                }
            }
            syn::Expr::Paren(_) => unimplemented!("is_tensor_expr::paren"),
            syn::Expr::Reference(reference) => { self.is_tensor_expr(&reference.expr) }
            syn::Expr::Try(try_expr) => {
                // println!("try_expr: {:#?}", try_expr.expr);
                self.is_tensor_expr(&try_expr.expr)
            }
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    self.variables
                        .get(ident)
                        .map(|(_, is_tensor)| *is_tensor)
                        .unwrap_or(false)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub(crate) fn is_tensor_ident(&self, ident: &syn::Ident) -> bool {
        self.variables
            .get(ident)
            .map(|(_, is_tensor)| *is_tensor)
            .unwrap_or(false)
    }
}

impl<'ast> Visit<'ast> for _Visitor<'ast> {
    fn visit_signature(&mut self, sig: &'ast syn::Signature) {
        for arg in sig.inputs.iter() {
            if let syn::FnArg::Typed(pat_type) = arg {
                let string = pat_type.ty.to_token_stream().to_string();
                let is_tensor = string.contains("Tensor") || string.contains("_Tensor");
                match pat_type.pat.as_ref() {
                    syn::Pat::Const(_) => unimplemented!("fuse_impl::const"),
                    syn::Pat::Ident(pat_ident) => {
                        let new_name = self.ssa_ctx.fresh_name(&pat_ident.ident.to_string());
                        if is_tensor {
                            self.nodes.push(
                                Node::Input(Var2 {
                                    ident: syn::Ident::new(&new_name, pat_ident.ident.span()),
                                })
                            );
                        }
                        self.declare_variable(pat_ident.ident.clone(), is_tensor);
                    }
                    syn::Pat::Lit(_) => unimplemented!("fuse_impl::lit"),
                    syn::Pat::Macro(_) => unimplemented!("fuse_impl::macro"),
                    syn::Pat::Or(_) => unimplemented!("fuse_impl::or"),
                    syn::Pat::Paren(_) => unimplemented!("fuse_impl::paren"),
                    syn::Pat::Path(_) => unimplemented!("fuse_impl::path"),
                    syn::Pat::Range(_) => unimplemented!("fuse_impl::range"),
                    syn::Pat::Reference(_) => unimplemented!("fuse_impl::reference"),
                    syn::Pat::Rest(_) => unimplemented!("fuse_impl::rest"),
                    syn::Pat::Slice(_) => unimplemented!("fuse_impl::slice"),
                    syn::Pat::Struct(_) => unimplemented!("fuse_impl::struct"),
                    syn::Pat::Tuple(_) => unimplemented!("fuse_impl::tuple"),
                    syn::Pat::TupleStruct(_) => unimplemented!("fuse_impl::tuple_struct"),
                    syn::Pat::Type(_) => unimplemented!("fuse_impl::type"),
                    syn::Pat::Verbatim(_) => unimplemented!("fuse_impl::verbatim"),
                    syn::Pat::Wild(_) => unimplemented!("fuse_impl::wild"),
                    _ => todo!(),
                }
            }
        }
        syn::visit::visit_signature(self, sig);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            match &local.pat {
                syn::Pat::Const(_) => todo!(),
                syn::Pat::Ident(pat_ident) => {
                    let ssa_name = self.ssa_ctx.fresh_name(&pat_ident.ident.to_string());
                    self.current_assignment = Some(
                        proc_macro2::Ident::new(&ssa_name, pat_ident.ident.span())
                    );
                    self.declare_variable(pat_ident.ident.clone(), self.is_tensor_expr(&init.expr));
                }
                syn::Pat::Lit(_) => todo!(),
                syn::Pat::Macro(_) => todo!(),
                syn::Pat::Or(_) => todo!(),
                syn::Pat::Paren(_) => todo!(),
                syn::Pat::Path(_) => todo!(),
                syn::Pat::Range(_) => todo!(),
                syn::Pat::Reference(_) => todo!(),
                syn::Pat::Rest(_) => todo!(),
                syn::Pat::Slice(_) => todo!(),
                syn::Pat::Struct(_) => todo!(),
                syn::Pat::Tuple(_) => todo!(),
                syn::Pat::TupleStruct(_) => todo!(),
                syn::Pat::Type(pat_type) => {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        let ssa_name = self.ssa_ctx.fresh_name(&pat_ident.ident.to_string());
                        self.current_assignment = Some(
                            proc_macro2::Ident::new(&ssa_name, pat_ident.ident.span())
                        );
                        self.declare_variable(
                            pat_ident.ident.clone(),
                            self.is_tensor_expr(&init.expr)
                        );
                    }
                }
                syn::Pat::Verbatim(_) => todo!(),
                syn::Pat::Wild(_) => todo!(),
                _ => todo!(),
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
            let out = self.ssa_ctx.fresh_name("__out");
            let out = proc_macro2::Ident::new(&out, node.span());
            self.declare_variable(out.clone(), false);
            self.mark_used(&out);
            out
        };
        let operand = self.ssa_ctx
            .current_name(&node.receiver.to_token_stream().to_string())
            .expect("not found");
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
            | "relu"
            | "selu" => {
                Node::Unary(Unary {
                    method: &node.method,
                    operand: proc_macro2::Ident::new(&operand, node.span()),
                    args: node.args
                        .iter()
                        .map(|arg| {
                            match arg {
                                syn::Expr::Path(expr_path) => {
                                    syn::Expr::Path(syn::ExprPath {
                                        attrs: expr_path.attrs.clone(),
                                        qself: expr_path.qself.clone(),
                                        path: {
                                            if expr_path.path.get_ident().is_some() {
                                                let mut path = expr_path.path.clone();
                                                path.segments[0].ident = syn::Ident::new(
                                                    &self.ssa_ctx
                                                        .current_name_expr(arg)
                                                        .expect(
                                                            format!(
                                                                "visit_expr_method_call::current_name_expr::{}",
                                                                arg.to_token_stream().to_string()
                                                            ).as_str()
                                                        )
                                                        .clone(),
                                                    expr_path.span()
                                                );
                                                path
                                            } else {
                                                expr_path.path.clone()
                                            }
                                        },
                                    })
                                }
                                _ => arg.clone(),
                            }
                        })
                        .collect(),
                    output: out.clone(),
                })
            }
            _ =>
                unimplemented!(
                    "_visitor::visit_expr_method_call::{}",
                    node.method.to_string().as_str()
                ),
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
        let left_var = syn::Ident::new(
            &self.ssa_ctx
                .current_name(&self.current_var.to_string())
                .expect(
                    format!(
                        "visit_expr_binary::current_name_expr::{}",
                        self.current_var.to_string()
                    ).as_str()
                )
                .clone(),
            i.left.span()
        );
        self.current_assignment = None;
        self.mark_expr_used(&i.right);
        self.visit_expr(&i.right);
        let right_var = syn::Ident::new(
            &self.ssa_ctx
                .current_name(&self.current_var.to_string())
                .expect(
                    format!(
                        "visit_expr_binary::current_name_expr::{}",
                        self.current_var.to_string()
                    ).as_str()
                )
                .clone(),
            i.right.span()
        );
        let out_is_tensor = self.is_tensor_expr(&i.left) || self.is_tensor_expr(&i.right);
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
            self.current_var = current_assignment.clone();
            current_assignment
        } else {
            let out = self.ssa_ctx.fresh_name("__out");
            self.current_var = proc_macro2::Ident::new("__out", i.span());
            let out = proc_macro2::Ident::new(&out, i.span());
            self.declare_variable(out.clone(), out_is_tensor);
            self.mark_used(&out);
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
    }
    fn visit_stmt_macro(&mut self, node: &'ast syn::StmtMacro) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        for token in node.mac.tokens.clone() {
            println!("{:#?}", token);
            match token {
                proc_macro2::TokenTree::Ident(ident) => {
                    self.mark_used(&ident);
                }
                _ => {}
            }
        }
        self.visit_macro(&node.mac);
    }
}
