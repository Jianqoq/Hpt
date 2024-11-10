use quote::ToTokens;
use syn::{ spanned::Spanned, visit::* };

use super::node::{ Binary, Node, Unary };

pub(crate) struct Visitor<'ast> {
    pub(crate) nodes: Vec<Node<'ast>>,
    pub(crate) var_cnt: usize,
    pub(crate) current_var: proc_macro2::Ident,
    pub(crate) current_assignment: Option<proc_macro2::Ident>,
}

impl<'ast> Visitor<'ast> {
    pub(crate) fn new() -> Self {
        use proc_macro2::Span;
        Self {
            nodes: vec![],
            var_cnt: 0,
            current_var: proc_macro2::Ident::new("__out0", Span::call_site()),
            current_assignment: None,
        }
    }
}

impl<'ast> Visit<'ast> for Visitor<'ast> {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            if let syn::Pat::Ident(syn::PatIdent { ident, .. }) = &local.pat {
                self.current_assignment = Some(ident.clone());
            }
            self.visit_expr(&init.expr);
            self.current_assignment = None;
        } else {
            unimplemented!("only support assignment for now");
        }
    }
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let mut outs: Vec<syn::Ident> = vec![];
        let current_assignment = self.current_assignment.clone();
        self.visit_expr(&node.receiver);
        for arg in &node.args {
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
            let out = proc_macro2::Ident::new(&format!("__out{}", self.var_cnt), node.span());
            self.current_var = out.clone();
            self.var_cnt += 1;
            out
        };
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
                Node::Unary(
                    Unary {
                        method: &node.method,
                        operand: proc_macro2::Ident::new(
                            &node.receiver.to_token_stream().to_string(),
                            node.span()
                        ),
                        output: out.clone(),
                    },
                    0
                )
            }
            _ => todo!(),
        };
        self.nodes.push(method);
        // println!("{:#?}", self.nodes);
    }

    fn visit_ident(&mut self, i: &'ast proc_macro2::Ident) {
        self.current_var = i.clone();
    }

    fn visit_expr_reference(&mut self, i: &'ast syn::ExprReference) {
        self.visit_expr(&i.expr);
    }

    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        if i.path.segments.len() == 1 {
            self.visit_ident(&i.path.segments[0].ident);
        } else {
            visit_expr_path(self, i);
        }
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        let current_assignment = self.current_assignment.clone();
        self.current_assignment = None;
        self.visit_expr(&i.left);
        let left_var = self.current_var.clone();
        self.current_assignment = None;
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
            let out = proc_macro2::Ident::new(&format!("__out{}", self.var_cnt), i.span());
            self.current_var = out.clone();
            self.var_cnt += 1;
            out
        };
        self.nodes.push(
            Node::Binary(
                Binary {
                    method: proc_macro2::Ident::new(method, i.span()),
                    left: left_var,
                    right: right_var,
                    output: out.clone(),
                },
                0
            )
        );
        self.current_var = out;
        self.var_cnt += 1;
    }
}
