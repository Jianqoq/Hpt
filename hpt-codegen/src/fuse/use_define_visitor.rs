use std::collections::HashSet;
use syn::{spanned::Spanned, visit::Visit};

use super::variable_collector::VariableCollector;
#[derive(Debug)]
pub(crate) enum State {
    Define,
    Use,
    Assign,
}

pub(crate) struct UseDefineVisitor {
    pub(crate) used_vars: HashSet<syn::Ident>,
    pub(crate) define_vars: HashSet<syn::Ident>,
    pub(crate) assigned_vars: HashSet<syn::Ident>,
    pub(crate) state: State,
}

impl UseDefineVisitor {
    pub(crate) fn new() -> Self {
        Self {
            used_vars: HashSet::new(),
            define_vars: HashSet::new(),
            assigned_vars: HashSet::new(),
            state: State::Define,
        }
    }

    pub(crate) fn insert(&mut self, ident: syn::Ident) -> bool {
        match self.state {
            State::Define => self.define_vars.insert(ident),
            State::Use => self.used_vars.insert(ident),
            State::Assign => self.assigned_vars.insert(ident),
        }
    }

    pub(crate) fn extend<I: IntoIterator<Item = syn::Ident>>(&mut self, iter: I) {
        match self.state {
            State::Define => self.define_vars.extend(iter),
            State::Use => self.used_vars.extend(iter),
            State::Assign => self.assigned_vars.extend(iter),
        }
    }
}

impl<'ast> Visit<'ast> for UseDefineVisitor {
    fn visit_pat(&mut self, pat: &'ast syn::Pat) {
        match pat {
            syn::Pat::Const(_) => unimplemented!("use_define_visitor::visit_pat::const"),
            syn::Pat::Ident(ident) => {
                self.insert(ident.ident.clone());
            }
            syn::Pat::Lit(_) => {}
            syn::Pat::Macro(_) => unimplemented!("use_define_visitor::visit_pat::macro"),
            syn::Pat::Or(_) => unimplemented!("use_define_visitor::visit_pat::or"),
            syn::Pat::Paren(paren) => {
                self.visit_pat(&paren.pat);
            }
            syn::Pat::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    self.insert(ident.clone());
                }
            }
            syn::Pat::Range(_) => unimplemented!("use_define_visitor::visit_pat::range"),
            syn::Pat::Reference(reference) => {
                self.visit_pat(&reference.pat);
            }
            syn::Pat::Rest(_) => {}
            syn::Pat::Slice(slice) => {
                for elem in slice.elems.iter() {
                    self.visit_pat(elem);
                }
            }
            syn::Pat::Struct(struct_pat) => {
                for field in struct_pat.fields.iter() {
                    self.visit_pat(&field.pat);
                }
            }
            syn::Pat::Tuple(_) => {
                let mut collector = VariableCollector::new();
                collector.visit_pat(pat);
                self.extend(collector.vars);
            }
            syn::Pat::TupleStruct(tuple_struct) => {
                for elem in tuple_struct.elems.iter() {
                    self.visit_pat(elem);
                }
            }
            syn::Pat::Type(ty) => self.visit_pat(&ty.pat),
            syn::Pat::Verbatim(_) => unimplemented!("use_define_visitor::visit_pat::verbatim"),
            syn::Pat::Wild(_) => {
                self.insert(syn::Ident::new("_", pat.span()));
            }
            _ => unimplemented!("use_define_visitor::visit_pat::other"),
        }
    }
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.state = State::Define;
        self.insert(i.sig.ident.clone());
    }
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        let mut collector = VariableCollector::new();
        self.state = State::Assign;
        if let syn::Expr::Path(left) = node.left.as_ref() {
            if let Some(ident) = left.path.get_ident() {
                self.insert(ident.clone());
            }
        }
        self.state = State::Use;
        collector.visit_expr(node.right.as_ref());
        self.extend(collector.vars);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        let mut collector = VariableCollector::new();
        self.state = State::Define;
        collector.visit_pat(&i.pat);
        self.extend(collector.vars);
        if let Some(init) = &i.init {
            let mut collector = VariableCollector::new();
            collector.visit_local_init(init);
            self.state = State::Use;
            self.extend(collector.vars);
        }
    }
    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        self.state = State::Use;
        if let syn::Expr::Path(left) = i.left.as_ref() {
            if let Some(ident) = left.path.get_ident() {
                self.insert(ident.clone());
            }
        } else {
            self.visit_expr(i.left.as_ref());
        }
        self.state = State::Use;
        if let syn::Expr::Path(right) = i.right.as_ref() {
            if let Some(ident) = right.path.get_ident() {
                self.insert(ident.clone());
            }
        } else {
            self.visit_expr(i.right.as_ref());
        }
    }
    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        self.state = State::Use;
        for arg in i.args.iter() {
            match arg {
                syn::Expr::Array(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::array")
                }
                syn::Expr::Assign(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::assign")
                }
                syn::Expr::Async(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::async")
                }
                syn::Expr::Await(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::await")
                }
                syn::Expr::Binary(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::binary")
                }
                syn::Expr::Block(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::block")
                }
                syn::Expr::Break(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::break")
                }
                syn::Expr::Call(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::call")
                }
                syn::Expr::Cast(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::cast")
                }
                syn::Expr::Closure(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::closure")
                }
                syn::Expr::Const(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::const")
                }
                syn::Expr::Continue(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::continue")
                }
                syn::Expr::Field(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::field")
                }
                syn::Expr::ForLoop(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::for_loop")
                }
                syn::Expr::Group(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::group")
                }
                syn::Expr::If(_) => unimplemented!("fuse::use_define_visitor::visit_expr_call::if"),
                syn::Expr::Index(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::index")
                }
                syn::Expr::Infer(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::infer")
                }
                syn::Expr::Let(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::let")
                }
                syn::Expr::Lit(_) => {}
                syn::Expr::Loop(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::loop")
                }
                syn::Expr::Macro(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::macro")
                }
                syn::Expr::Match(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::match")
                }
                syn::Expr::MethodCall(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::method_call")
                }
                syn::Expr::Paren(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::paren")
                }
                syn::Expr::Path(path) => {
                    if let Some(ident) = path.path.get_ident() {
                        self.used_vars.insert(ident.clone());
                    }
                }
                syn::Expr::Range(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::range")
                }
                syn::Expr::RawAddr(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::raw_addr")
                }
                syn::Expr::Reference(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::reference")
                }
                syn::Expr::Repeat(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::repeat")
                }
                syn::Expr::Return(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::return")
                }
                syn::Expr::Struct(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::struct")
                }
                syn::Expr::Try(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::try")
                }
                syn::Expr::TryBlock(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::try_block")
                }
                syn::Expr::Tuple(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::tuple")
                }
                syn::Expr::Unary(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::unary")
                }
                syn::Expr::Unsafe(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::unsafe")
                }
                syn::Expr::Verbatim(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::verbatim")
                }
                syn::Expr::While(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::while")
                }
                syn::Expr::Yield(_) => {
                    unimplemented!("fuse::use_define_visitor::visit_expr_call::yield")
                }
                _ => unimplemented!("fuse::use_define_visitor::visit_expr_call::other"),
            }
        }
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        self.state = State::Use;
        if let syn::Expr::Path(path) = i.receiver.as_ref() {
            if let Some(ident) = path.path.get_ident() {
                self.insert(ident.clone());
            }
        } else {
            self.visit_expr(i.receiver.as_ref());
        }
        for arg in i.args.iter() {
            self.state = State::Use;
            if let syn::Expr::Path(path) = arg {
                if let Some(ident) = path.path.get_ident() {
                    self.insert(ident.clone());
                }
            } else {
                self.visit_expr(arg);
            }
        }
    }

    fn visit_expr_let(&mut self, i: &'ast syn::ExprLet) {
        self.state = State::Define;
        self.visit_pat(&i.pat);
        self.state = State::Use;
        self.visit_expr(&i.expr);
    }
}
