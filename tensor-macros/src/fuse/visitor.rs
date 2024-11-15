use std::collections::HashMap;

use quote::ToTokens;
use syn::{ spanned::Spanned, visit::* };

use super::{ node::{ Binary, Node, Unary }, rcmut::RCMut, ssa::SSAContext };

#[derive(Clone)]
pub(crate) struct Variables {
    pub(crate) vars: HashMap<syn::Ident, (bool, bool)>,
    pub(crate) prev_vars: Option<RCMut<Variables>>,
}

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
    pub(crate) current_type: Option<syn::Type>,
    pub(crate) variables: RCMut<Variables>,
    pub(crate) next_visitor: Option<Box<_Visitor<'ast>>>,
    pub(crate) ssa_ctx: RCMut<SSAContext>,
    pub(crate) errors: Vec<syn::Error>,
}

impl<'ast> _Visitor<'ast> {
    pub(crate) fn new() -> Self {
        use proc_macro2::Span;
        Self {
            nodes: vec![],
            current_var: proc_macro2::Ident::new("__out0", Span::call_site()),
            current_assignment: None,
            current_type: None,
            variables: RCMut::new(Variables {
                vars: HashMap::new(),
                prev_vars: None,
            }),
            next_visitor: None,
            ssa_ctx: RCMut::new(SSAContext::new()),
            errors: vec![],
        }
    }
    pub(crate) fn declare_variable(&mut self, ident: syn::Ident, is_tensor: bool) {
        self.variables.borrow_mut().vars.insert(ident, (false, is_tensor));
    }
    pub(crate) fn mark_used(&mut self, name: &syn::Ident) {
        let mut prev_vars = Some(self.variables.clone());
        let mut updated = false;
        while !updated {
            if let Some(prev) = prev_vars {
                if let Some((usage, _)) = prev.borrow_mut().vars.get_mut(name) {
                    *usage = true;
                    updated = true;
                }
                prev_vars = prev.borrow().prev_vars.clone();
            }
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
        for (ident, (usage, _)) in self.variables.borrow().vars.iter() {
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
            syn::Expr::Try(try_expr) => { self.is_tensor_expr(&try_expr.expr) }
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    let mut current_scope = self.variables
                        .borrow()
                        .vars.get(ident)
                        .map(|(_, is_tensor)| *is_tensor)
                        .unwrap_or(false);
                    let mut prev_vars = Some(self.variables.clone());
                    while let Some(prev) = prev_vars {
                        if let Some((_, is_tensor)) = prev.borrow().vars.get(ident) {
                            current_scope |= *is_tensor;
                        }
                        prev_vars = prev.borrow().prev_vars.clone();
                    }
                    current_scope
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub(crate) fn process_expr_method_call_args(
        &mut self,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>
    ) -> Vec<syn::Expr> {
        args.iter()
            .map(|arg| {
                match arg {
                    syn::Expr::Path(expr_path) => {
                        let mut expr_path = expr_path.clone();
                        expr_path.path = if expr_path.path.get_ident().is_some() {
                            let mut path = expr_path.path.clone();
                            let ctx = self.ssa_ctx.borrow();
                            let current_name = &ctx
                                .current_name_expr(arg)
                                .ok_or_else(|| {
                                    syn::Error::new(
                                        arg.span(),
                                        format!(
                                            "visit_expr_method_call::current_name_expr: Cannot find {} in current scope.",
                                            arg.to_token_stream().to_string()
                                        )
                                    )
                                });
                            if let Ok(name) = current_name {
                                path.segments[0].ident = syn::Ident::new(&name, arg.span());
                            } else {
                                self.errors.push(current_name.as_ref().unwrap_err().clone());
                            }
                            path
                        } else {
                            expr_path.path.clone()
                        };
                        syn::Expr::Path(expr_path)
                    }
                    _ => arg.clone(),
                }
            })
            .collect()
    }
}

impl<'ast> Visit<'ast> for _Visitor<'ast> {
    fn visit_abi(&mut self, i: &'ast syn::Abi) {
        unimplemented!("visitor::visit_abi");
    }

    fn visit_angle_bracketed_generic_arguments(
        &mut self,
        i: &'ast syn::AngleBracketedGenericArguments
    ) {
        unimplemented!("visitor::visit_angle_bracketed_generic_arguments");
    }

    fn visit_arm(&mut self, i: &'ast syn::Arm) {
        unimplemented!("visitor::visit_arm");
    }

    fn visit_assoc_const(&mut self, i: &'ast syn::AssocConst) {
        unimplemented!("visitor::visit_assoc_const");
    }

    fn visit_assoc_type(&mut self, i: &'ast syn::AssocType) {
        unimplemented!("visitor::visit_assoc_type");
    }

    fn visit_attr_style(&mut self, i: &'ast syn::AttrStyle) {
        unimplemented!("visitor::visit_attr_style");
    }

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        unimplemented!("visitor::visit_attribute");
    }

    fn visit_bare_fn_arg(&mut self, i: &'ast syn::BareFnArg) {
        unimplemented!("visitor::visit_bare_fn_arg");
    }

    fn visit_bare_variadic(&mut self, i: &'ast syn::BareVariadic) {
        unimplemented!("visitor::visit_bare_variadic");
    }

    fn visit_bin_op(&mut self, i: &'ast syn::BinOp) {
        unimplemented!("visitor::visit_bin_op");
    }

    fn visit_bound_lifetimes(&mut self, i: &'ast syn::BoundLifetimes) {
        unimplemented!("visitor::visit_bound_lifetimes");
    }

    fn visit_captured_param(&mut self, i: &'ast syn::CapturedParam) {
        unimplemented!("visitor::visit_captured_param");
    }

    fn visit_const_param(&mut self, i: &'ast syn::ConstParam) {
        unimplemented!("visitor::visit_const_param");
    }

    fn visit_constraint(&mut self, i: &'ast syn::Constraint) {
        unimplemented!("visitor::visit_constraint");
    }

    fn visit_data(&mut self, i: &'ast syn::Data) {
        unimplemented!("visitor::visit_data");
    }

    fn visit_data_enum(&mut self, i: &'ast syn::DataEnum) {
        unimplemented!("visitor::visit_data_enum");
    }

    fn visit_data_struct(&mut self, i: &'ast syn::DataStruct) {
        unimplemented!("visitor::visit_data_struct");
    }

    fn visit_data_union(&mut self, i: &'ast syn::DataUnion) {
        unimplemented!("visitor::visit_data_union");
    }

    fn visit_derive_input(&mut self, i: &'ast syn::DeriveInput) {
        unimplemented!("visitor::visit_derive_input");
    }

    fn visit_expr_array(&mut self, i: &'ast syn::ExprArray) {
        unimplemented!("visitor::visit_expr_array");
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        unimplemented!("visitor::visit_expr_assign");
    }

    fn visit_expr_async(&mut self, i: &'ast syn::ExprAsync) {
        unimplemented!("visitor::visit_expr_async");
    }

    fn visit_expr_await(&mut self, i: &'ast syn::ExprAwait) {
        unimplemented!("visitor::visit_expr_await");
    }

    fn visit_expr_block(&mut self, i: &'ast syn::ExprBlock) {
        let mut next_visitor = _Visitor::new();
        next_visitor.variables.borrow_mut().prev_vars = Some(self.variables.clone());
        next_visitor.ssa_ctx.borrow_mut().prev_ssa_ctx = Some(self.ssa_ctx.clone());
        visit_expr_block(&mut next_visitor, i);
        self.next_visitor = Some(Box::new(next_visitor));
    }

    fn visit_expr_break(&mut self, i: &'ast syn::ExprBreak) {
        unimplemented!("visitor::visit_expr_break");
    }

    fn visit_expr_cast(&mut self, i: &'ast syn::ExprCast) {
        unimplemented!("visitor::visit_expr_cast");
    }

    fn visit_expr_closure(&mut self, i: &'ast syn::ExprClosure) {
        unimplemented!("visitor::visit_expr_closure");
    }

    fn visit_expr_const(&mut self, i: &'ast syn::ExprConst) {
        unimplemented!("visitor::visit_expr_const");
    }

    fn visit_expr_continue(&mut self, i: &'ast syn::ExprContinue) {
        unimplemented!("visitor::visit_expr_continue");
    }

    fn visit_expr_field(&mut self, i: &'ast syn::ExprField) {
        unimplemented!("visitor::visit_expr_field");
    }

    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        unimplemented!("visitor::visit_expr_for_loop");
    }

    fn visit_expr_group(&mut self, i: &'ast syn::ExprGroup) {
        unimplemented!("visitor::visit_expr_group");
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        unimplemented!("visitor::visit_expr_index");
    }

    fn visit_expr_infer(&mut self, i: &'ast syn::ExprInfer) {
        unimplemented!("visitor::visit_expr_infer");
    }

    fn visit_expr_let(&mut self, i: &'ast syn::ExprLet) {
        unimplemented!("visitor::visit_expr_let");
    }

    fn visit_expr_loop(&mut self, i: &'ast syn::ExprLoop) {
        unimplemented!("visitor::visit_expr_loop");
    }

    fn visit_expr_macro(&mut self, i: &'ast syn::ExprMacro) {
        unimplemented!("visitor::visit_expr_macro");
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        unimplemented!("visitor::visit_expr_match");
    }

    fn visit_expr_paren(&mut self, i: &'ast syn::ExprParen) {
        unimplemented!("visitor::visit_expr_paren");
    }

    fn visit_expr_range(&mut self, i: &'ast syn::ExprRange) {
        unimplemented!("visitor::visit_expr_range");
    }

    fn visit_expr_raw_addr(&mut self, i: &'ast syn::ExprRawAddr) {
        unimplemented!("visitor::visit_expr_raw_addr");
    }

    fn visit_expr_repeat(&mut self, i: &'ast syn::ExprRepeat) {
        unimplemented!("visitor::visit_expr_repeat");
    }

    fn visit_expr_return(&mut self, i: &'ast syn::ExprReturn) {
        unimplemented!("visitor::visit_expr_return");
    }

    fn visit_expr_struct(&mut self, i: &'ast syn::ExprStruct) {
        unimplemented!("visitor::visit_expr_struct");
    }

    fn visit_expr_try_block(&mut self, i: &'ast syn::ExprTryBlock) {
        unimplemented!("visitor::visit_expr_try_block");
    }

    fn visit_expr_unary(&mut self, i: &'ast syn::ExprUnary) {
        unimplemented!("visitor::visit_expr_unary");
    }

    fn visit_expr_unsafe(&mut self, i: &'ast syn::ExprUnsafe) {
        unimplemented!("visitor::visit_expr_unsafe");
    }

    fn visit_expr_while(&mut self, i: &'ast syn::ExprWhile) {
        unimplemented!("visitor::visit_expr_while");
    }

    fn visit_expr_yield(&mut self, i: &'ast syn::ExprYield) {
        unimplemented!("visitor::visit_expr_yield");
    }

    fn visit_field(&mut self, i: &'ast syn::Field) {
        unimplemented!("visitor::visit_field");
    }

    fn visit_field_mutability(&mut self, i: &'ast syn::FieldMutability) {
        unimplemented!("visitor::visit_field_mutability");
    }

    fn visit_field_pat(&mut self, i: &'ast syn::FieldPat) {
        unimplemented!("visitor::visit_field_pat");
    }

    fn visit_field_value(&mut self, i: &'ast syn::FieldValue) {
        unimplemented!("visitor::visit_field_value");
    }

    fn visit_fields(&mut self, i: &'ast syn::Fields) {
        unimplemented!("visitor::visit_fields");
    }

    fn visit_fields_named(&mut self, i: &'ast syn::FieldsNamed) {
        unimplemented!("visitor::visit_fields_named");
    }

    fn visit_fields_unnamed(&mut self, i: &'ast syn::FieldsUnnamed) {
        unimplemented!("visitor::visit_fields_unnamed");
    }

    fn visit_file(&mut self, i: &'ast syn::File) {
        unimplemented!("visitor::visit_file");
    }

    fn visit_foreign_item(&mut self, i: &'ast syn::ForeignItem) {
        unimplemented!("visitor::visit_foreign_item");
    }

    fn visit_foreign_item_fn(&mut self, i: &'ast syn::ForeignItemFn) {
        unimplemented!("visitor::visit_foreign_item_fn");
    }

    fn visit_foreign_item_macro(&mut self, i: &'ast syn::ForeignItemMacro) {
        unimplemented!("visitor::visit_foreign_item_macro");
    }

    fn visit_foreign_item_static(&mut self, i: &'ast syn::ForeignItemStatic) {
        unimplemented!("visitor::visit_foreign_item_static");
    }

    fn visit_foreign_item_type(&mut self, i: &'ast syn::ForeignItemType) {
        unimplemented!("visitor::visit_foreign_item_type");
    }

    fn visit_generic_argument(&mut self, i: &'ast syn::GenericArgument) {
        unimplemented!("visitor::visit_generic_argument");
    }

    fn visit_generic_param(&mut self, i: &'ast syn::GenericParam) {
        unimplemented!("visitor::visit_generic_param");
    }

    fn visit_impl_item(&mut self, i: &'ast syn::ImplItem) {
        unimplemented!("visitor::visit_impl_item");
    }

    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        unimplemented!("visitor::visit_impl_item_const");
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        unimplemented!("visitor::visit_impl_item_fn");
    }

    fn visit_impl_item_macro(&mut self, i: &'ast syn::ImplItemMacro) {
        unimplemented!("visitor::visit_impl_item_macro");
    }

    fn visit_impl_item_type(&mut self, i: &'ast syn::ImplItemType) {
        unimplemented!("visitor::visit_impl_item_type");
    }

    fn visit_impl_restriction(&mut self, i: &'ast syn::ImplRestriction) {
        unimplemented!("visitor::visit_impl_restriction");
    }

    fn visit_index(&mut self, i: &'ast syn::Index) {
        unimplemented!("visitor::visit_index");
    }

    fn visit_item(&mut self, i: &'ast syn::Item) {
        unimplemented!("visitor::visit_item");
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        unimplemented!("visitor::visit_item_const");
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        unimplemented!("visitor::visit_item_enum");
    }

    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        unimplemented!("visitor::visit_item_extern_crate");
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        unimplemented!("visitor::visit_item_fn");
    }

    fn visit_item_foreign_mod(&mut self, i: &'ast syn::ItemForeignMod) {
        unimplemented!("visitor::visit_item_foreign_mod");
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        unimplemented!("visitor::visit_item_impl");
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        unimplemented!("visitor::visit_item_macro");
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        unimplemented!("visitor::visit_item_mod");
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        unimplemented!("visitor::visit_item_static");
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        unimplemented!("visitor::visit_item_struct");
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        unimplemented!("visitor::visit_item_trait");
    }

    fn visit_item_trait_alias(&mut self, i: &'ast syn::ItemTraitAlias) {
        unimplemented!("visitor::visit_item_trait_alias");
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        unimplemented!("visitor::visit_item_type");
    }

    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        unimplemented!("visitor::visit_item_union");
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        unimplemented!("visitor::visit_item_use");
    }

    fn visit_label(&mut self, i: &'ast syn::Label) {
        unimplemented!("visitor::visit_label");
    }

    fn visit_macro(&mut self, i: &'ast syn::Macro) {
        unimplemented!("visitor::visit_macro");
    }

    fn visit_macro_delimiter(&mut self, i: &'ast syn::MacroDelimiter) {
        unimplemented!("visitor::visit_macro_delimiter");
    }

    fn visit_member(&mut self, i: &'ast syn::Member) {
        unimplemented!("visitor::visit_member");
    }

    fn visit_meta(&mut self, i: &'ast syn::Meta) {
        unimplemented!("visitor::visit_meta");
    }

    fn visit_meta_list(&mut self, i: &'ast syn::MetaList) {
        unimplemented!("visitor::visit_meta_list");
    }

    fn visit_meta_name_value(&mut self, i: &'ast syn::MetaNameValue) {
        unimplemented!("visitor::visit_meta_name_value");
    }

    fn visit_parenthesized_generic_arguments(
        &mut self,
        i: &'ast syn::ParenthesizedGenericArguments
    ) {
        unimplemented!("visitor::visit_parenthesized_generic_arguments");
    }

    fn visit_pat_or(&mut self, i: &'ast syn::PatOr) {
        unimplemented!("visitor::visit_pat_or");
    }

    fn visit_pat_paren(&mut self, i: &'ast syn::PatParen) {
        unimplemented!("visitor::visit_pat_paren");
    }

    fn visit_pat_reference(&mut self, i: &'ast syn::PatReference) {
        unimplemented!("visitor::visit_pat_reference");
    }

    fn visit_pat_rest(&mut self, i: &'ast syn::PatRest) {
        unimplemented!("visitor::visit_pat_rest");
    }

    fn visit_pat_slice(&mut self, i: &'ast syn::PatSlice) {
        unimplemented!("visitor::visit_pat_slice");
    }

    fn visit_pat_struct(&mut self, i: &'ast syn::PatStruct) {
        unimplemented!("visitor::visit_pat_struct");
    }

    fn visit_pat_tuple(&mut self, i: &'ast syn::PatTuple) {
        unimplemented!("visitor::visit_pat_tuple");
    }

    fn visit_pat_tuple_struct(&mut self, i: &'ast syn::PatTupleStruct) {
        unimplemented!("visitor::visit_pat_tuple_struct");
    }

    fn visit_pat_type(&mut self, node: &'ast syn::PatType) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        match node.pat.as_ref() {
            syn::Pat::Const(..) => unimplemented!("visitor::visit_pat_type::Const"),
            syn::Pat::Ident(pat_ident) => {
                self.visit_pat_ident(pat_ident);
                let ty = node.ty.to_token_stream().to_string();
                let is_tensor = ty.contains("Tensor") || ty.contains("_Tensor");
                self.ssa_ctx.borrow_mut().fresh_name(&self.current_var.to_string());
                self.declare_variable(self.current_var.clone(), is_tensor);
            }
            syn::Pat::Lit(_) => unimplemented!("visitor::visit_pat_type::Lit"),
            syn::Pat::Macro(_) => unimplemented!("visitor::visit_pat_type::Macro"),
            syn::Pat::Or(_) => unimplemented!("visitor::visit_pat_type::Or"),
            syn::Pat::Paren(_) => unimplemented!("visitor::visit_pat_type::Paren"),
            syn::Pat::Path(_) => unimplemented!("visitor::visit_pat_type::Path"),
            syn::Pat::Range(_) => unimplemented!("visitor::visit_pat_type::Range"),
            syn::Pat::Reference(_) => unimplemented!("visitor::visit_pat_type::Reference"),
            syn::Pat::Rest(_) => unimplemented!("visitor::visit_pat_type::Rest"),
            syn::Pat::Slice(_) => unimplemented!("visitor::visit_pat_type::Slice"),
            syn::Pat::Struct(_) => unimplemented!("visitor::visit_pat_type::Struct"),
            syn::Pat::Tuple(_) => unimplemented!("visitor::visit_pat_type::Tuple"),
            syn::Pat::TupleStruct(_) => unimplemented!("visitor::visit_pat_type::TupleStruct"),
            syn::Pat::Type(_) => unimplemented!("visitor::visit_pat_type::Type"),
            syn::Pat::Verbatim(_) => unimplemented!("visitor::visit_pat_type::Verbatim"),
            syn::Pat::Wild(_) => unimplemented!("visitor::visit_pat_type::Wild"),
            _ => unimplemented!("visitor::visit_pat_type::Other"),
        }
    }

    fn visit_pat_wild(&mut self, i: &'ast syn::PatWild) {
        unimplemented!("visitor::visit_pat_wild");
    }

    fn visit_path(&mut self, i: &'ast syn::Path) {
        unimplemented!("visitor::visit_path");
    }

    fn visit_path_arguments(&mut self, i: &'ast syn::PathArguments) {
        unimplemented!("visitor::visit_path_arguments");
    }

    fn visit_path_segment(&mut self, i: &'ast syn::PathSegment) {
        unimplemented!("visitor::visit_path_segment");
    }

    fn visit_pointer_mutability(&mut self, i: &'ast syn::PointerMutability) {
        unimplemented!("visitor::visit_pointer_mutability");
    }

    fn visit_precise_capture(&mut self, i: &'ast syn::PreciseCapture) {
        unimplemented!("visitor::visit_precise_capture");
    }

    fn visit_predicate_lifetime(&mut self, i: &'ast syn::PredicateLifetime) {
        unimplemented!("visitor::visit_predicate_lifetime");
    }

    fn visit_predicate_type(&mut self, i: &'ast syn::PredicateType) {
        unimplemented!("visitor::visit_predicate_type");
    }

    fn visit_qself(&mut self, i: &'ast syn::QSelf) {
        unimplemented!("visitor::visit_qself");
    }

    fn visit_range_limits(&mut self, i: &'ast syn::RangeLimits) {
        unimplemented!("visitor::visit_range_limits");
    }

    fn visit_receiver(&mut self, i: &'ast syn::Receiver) {
        unimplemented!("visitor::visit_receiver");
    }

    fn visit_span(&mut self, i: &proc_macro2::Span) {
        unimplemented!("visitor::visit_span");
    }

    fn visit_static_mutability(&mut self, i: &'ast syn::StaticMutability) {
        unimplemented!("visitor::visit_static_mutability");
    }

    fn visit_trait_bound(&mut self, i: &'ast syn::TraitBound) {
        unimplemented!("visitor::visit_trait_bound");
    }

    fn visit_trait_bound_modifier(&mut self, i: &'ast syn::TraitBoundModifier) {
        unimplemented!("visitor::visit_trait_bound_modifier");
    }

    fn visit_trait_item(&mut self, i: &'ast syn::TraitItem) {
        unimplemented!("visitor::visit_trait_item");
    }

    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        unimplemented!("visitor::visit_trait_item_const");
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        unimplemented!("visitor::visit_trait_item_fn");
    }

    fn visit_trait_item_macro(&mut self, i: &'ast syn::TraitItemMacro) {
        unimplemented!("visitor::visit_trait_item_macro");
    }

    fn visit_trait_item_type(&mut self, i: &'ast syn::TraitItemType) {
        unimplemented!("visitor::visit_trait_item_type");
    }

    fn visit_type(&mut self, i: &'ast syn::Type) {
        self.current_type = Some(i.clone());
    }

    fn visit_type_array(&mut self, i: &'ast syn::TypeArray) {
        unimplemented!("visitor::visit_type_array");
    }

    fn visit_type_bare_fn(&mut self, i: &'ast syn::TypeBareFn) {
        unimplemented!("visitor::visit_type_bare_fn");
    }

    fn visit_type_group(&mut self, i: &'ast syn::TypeGroup) {
        unimplemented!("visitor::visit_type_group");
    }

    fn visit_type_impl_trait(&mut self, i: &'ast syn::TypeImplTrait) {
        unimplemented!("visitor::visit_type_impl_trait");
    }

    fn visit_type_infer(&mut self, i: &'ast syn::TypeInfer) {
        unimplemented!("visitor::visit_type_infer");
    }

    fn visit_type_macro(&mut self, i: &'ast syn::TypeMacro) {
        unimplemented!("visitor::visit_type_macro");
    }

    fn visit_type_never(&mut self, i: &'ast syn::TypeNever) {
        unimplemented!("visitor::visit_type_never");
    }

    fn visit_type_param(&mut self, i: &'ast syn::TypeParam) {
        unimplemented!("visitor::visit_type_param");
    }

    fn visit_type_param_bound(&mut self, i: &'ast syn::TypeParamBound) {
        unimplemented!("visitor::visit_type_param_bound");
    }

    fn visit_type_paren(&mut self, i: &'ast syn::TypeParen) {
        unimplemented!("visitor::visit_type_paren");
    }

    fn visit_type_path(&mut self, i: &'ast syn::TypePath) {
        unimplemented!("visitor::visit_type_path");
    }

    fn visit_type_ptr(&mut self, i: &'ast syn::TypePtr) {
        unimplemented!("visitor::visit_type_ptr");
    }

    fn visit_type_reference(&mut self, i: &'ast syn::TypeReference) {
        unimplemented!("visitor::visit_type_reference");
    }

    fn visit_type_slice(&mut self, i: &'ast syn::TypeSlice) {
        unimplemented!("visitor::visit_type_slice");
    }

    fn visit_type_trait_object(&mut self, i: &'ast syn::TypeTraitObject) {
        unimplemented!("visitor::visit_type_trait_object");
    }

    fn visit_type_tuple(&mut self, i: &'ast syn::TypeTuple) {
        unimplemented!("visitor::visit_type_tuple");
    }

    fn visit_un_op(&mut self, i: &'ast syn::UnOp) {
        unimplemented!("visitor::visit_un_op");
    }

    fn visit_use_glob(&mut self, i: &'ast syn::UseGlob) {
        unimplemented!("visitor::visit_use_glob");
    }

    fn visit_use_group(&mut self, i: &'ast syn::UseGroup) {
        unimplemented!("visitor::visit_use_group");
    }

    fn visit_use_name(&mut self, i: &'ast syn::UseName) {
        unimplemented!("visitor::visit_use_name");
    }

    fn visit_use_path(&mut self, i: &'ast syn::UsePath) {
        unimplemented!("visitor::visit_use_path");
    }

    fn visit_use_rename(&mut self, i: &'ast syn::UseRename) {
        unimplemented!("visitor::visit_use_rename");
    }

    fn visit_use_tree(&mut self, i: &'ast syn::UseTree) {
        unimplemented!("visitor::visit_use_tree");
    }

    fn visit_variadic(&mut self, i: &'ast syn::Variadic) {
        unimplemented!("visitor::visit_variadic");
    }

    fn visit_variant(&mut self, i: &'ast syn::Variant) {
        unimplemented!("visitor::visit_variant");
    }

    fn visit_vis_restricted(&mut self, i: &'ast syn::VisRestricted) {
        unimplemented!("visitor::visit_vis_restricted");
    }

    fn visit_where_clause(&mut self, i: &'ast syn::WhereClause) {
        unimplemented!("visitor::visit_where_clause");
    }

    fn visit_where_predicate(&mut self, i: &'ast syn::WherePredicate) {
        unimplemented!("visitor::visit_where_predicate");
    }
    fn visit_local(&mut self, local: &'ast syn::Local) {
        for it in &local.attrs {
            self.visit_attribute(it);
        }
        match &local.pat {
            syn::Pat::Ident(pat_ident) => {
                self.visit_pat_ident(pat_ident);
                self.ssa_ctx.borrow_mut().fresh_name(&pat_ident.ident.to_string());
                if let Some(it) = &local.init {
                    self.visit_local_init(it);
                    self.declare_variable(pat_ident.ident.clone(), self.is_tensor_expr(&it.expr));
                }
            }
            _ => {
                syn::visit::visit_pat(self, &local.pat);
                if let Some(it) = &local.init {
                    self.visit_local_init(it);
                }
            }
        }
    }
    fn visit_block(&mut self, i: &'ast syn::Block) {
        let mut next_visitor = _Visitor::new();
        next_visitor.variables.borrow_mut().prev_vars = Some(self.variables.clone());
        next_visitor.ssa_ctx.borrow_mut().prev_ssa_ctx = Some(self.ssa_ctx.clone());
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
            let out = self.ssa_ctx.borrow_mut().fresh_name("__out");
            let out = proc_macro2::Ident::new(&out, node.span());
            self.declare_variable(out.clone(), false);
            self.mark_used(&out);
            out
        };
        let operand = self.ssa_ctx
            .borrow()
            .current_name(&node.receiver.to_token_stream().to_string())
            .expect("not found")
            .clone();
        let args = match node.method.to_string().as_str() {
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
                vec![]
            }
            "selu" => {
                for i in node.args.iter() {
                    self.mark_expr_used(i);
                    if self.is_tensor_expr(i) {
                        self.errors.push(syn::Error::new(i.span(), "selu only accept scalar"));
                    }
                }
                self.process_expr_method_call_args(&node.args)
            }
            _ =>
                unimplemented!(
                    "_visitor::visit_expr_method_call::{}",
                    node.method.to_string().as_str()
                ),
        };
        let method = Node::Unary(Unary {
            method: &node.method,
            operand: proc_macro2::Ident::new(&operand, node.span()),
            args,
            output: out.clone(),
        });
        self.nodes.push(method);
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
        self.mark_used(i);
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
        self.visit_expr(&i.left);
        let left_var = syn::Ident::new(
            &self.ssa_ctx
                .borrow()
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
        if !self.variables.borrow().vars.contains_key(&self.current_var) {
            let current_var = self.current_var.clone();
            self.declare_variable(current_var.clone(), self.is_tensor_expr(&i.left));
            self.mark_used(&current_var);
            let ssa_name = self.ssa_ctx
                .borrow_mut()
                .current_name(&current_var.to_string())
                .expect("not found")
                .clone();
            let ssa_ident = syn::Ident::new(&ssa_name, i.right.span());
            let contains = self.nodes.iter().any(|node| {
                match node {
                    Node::Input(ident) => ident == &ssa_ident,
                    Node::Binary(binary) => binary.output == ssa_ident,
                    Node::Unary(unary) => unary.output == ssa_ident,
                }
            });
            if !contains {
                let node = Node::Input(ssa_ident);
                self.nodes.push(node);
            }
        }
        self.visit_expr(&i.right);
        let right_var = syn::Ident::new(
            &self.ssa_ctx
                .borrow()
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
        if !self.variables.borrow().vars.contains_key(&self.current_var) {
            let current_var = self.current_var.clone();
            self.declare_variable(current_var.clone(), self.is_tensor_expr(&i.left));
            self.mark_used(&current_var);
            let ssa_name = self.ssa_ctx
                .borrow_mut()
                .current_name(&current_var.to_string())
                .expect("not found")
                .clone();
            let ssa_ident = syn::Ident::new(&ssa_name, i.right.span());
            let contains = self.nodes.iter().any(|node| {
                match node {
                    Node::Input(ident) => ident == &ssa_ident,
                    Node::Binary(binary) => binary.output == ssa_ident,
                    Node::Unary(unary) => unary.output == ssa_ident,
                }
            });
            if !contains {
                let node = Node::Input(ssa_ident);
                self.nodes.push(node);
            }
        }
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
            let out = self.ssa_ctx.borrow_mut().fresh_name("__out");
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
