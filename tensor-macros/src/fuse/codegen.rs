use std::collections::{ HashMap, HashSet };
use quote::ToTokens;
use syn::visit::*;
use crate::TokenStream2;

use super::{ rcmut::RCMut, ssa::SSAContext, visitor::_Visitor };

pub(crate) struct Codegen<'ast> {
    pub(crate) _codegen: _Codegen<'ast>,
}

impl<'ast> Codegen<'ast> {
    pub(crate) fn get_code(&mut self) -> TokenStream2 {
        let mut token_stream = TokenStream2::new();
        token_stream.extend(self._codegen.get_code());
        if let Some(next_codegen) = &mut self._codegen.next_codegen {
            token_stream.extend(next_codegen.get_code());
        }
        token_stream
    }
}

impl<'ast> Visit<'ast> for Codegen<'ast> {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        for it in &node.attrs {
            self._codegen.visit_attribute(it);
        }
        self._codegen.push_tokens(node.vis.to_token_stream());
        self._codegen.visit_signature(&node.sig);
        let mut current_token = self._codegen.current_tokens.drain(..).collect::<Vec<_>>();
        syn::visit::visit_block(&mut self._codegen, &*node.block);
        let block_token = self._codegen.current_tokens.drain(..).collect::<TokenStream2>();
        current_token.push(quote::quote! { {#block_token} });
        self._codegen.current_tokens = current_token;
    }
}

pub(crate) struct _Codegen<'ast> {
    pub(crate) fused_codes: &'ast HashMap<syn::Ident, TokenStream2>,
    pub(crate) to_remove: &'ast Vec<HashSet<syn::Ident>>,
    pub(crate) current_tokens: Vec<TokenStream2>,
    pub(crate) ssa_ctx: RCMut<SSAContext>,
    pub(crate) _visitor: Option<&'ast Box<_Visitor<'ast>>>,
    pub(crate) next_codegen: Option<Box<_Codegen<'ast>>>,
    pub(crate) pat_ident_need_remove: bool,
    pub(crate) pat_ident_is_ret: bool,
}

impl<'ast> _Codegen<'ast> {
    fn push_tokens(&mut self, tokens: TokenStream2) {
        self.current_tokens.push(tokens);
    }

    pub(crate) fn get_code(&mut self) -> TokenStream2 {
        self.current_tokens.drain(..).collect::<TokenStream2>()
    }
}

impl<'ast> Visit<'ast> for _Codegen<'ast> {
    fn visit_signature(&mut self, node: &'ast syn::Signature) {
        if let Some(it) = &node.constness {
            self.push_tokens(it.to_token_stream());
        }
        if let Some(it) = &node.asyncness {
            self.push_tokens(it.to_token_stream());
        }
        if let Some(it) = &node.unsafety {
            self.push_tokens(it.to_token_stream());
        }
        if let Some(it) = &node.abi {
            self.visit_abi(it);
        }
        self.push_tokens(node.fn_token.to_token_stream());
        self.push_tokens(node.ident.to_token_stream());
        self.visit_generics(&node.generics);
        let mut current_token = self.current_tokens.drain(..).collect::<Vec<_>>();
        for el in syn::punctuated::Punctuated::pairs(&node.inputs) {
            let it = el.value();
            self.visit_fn_arg(it);
            if let Some(comma) = el.punct() {
                self.push_tokens(comma.to_token_stream());
            }
        }
        let arg_token = self.current_tokens.drain(..).collect::<TokenStream2>();
        current_token.push(quote::quote! { (#arg_token) });
        self.current_tokens = current_token;

        if let Some(it) = &node.variadic {
            self.visit_variadic(it);
        }
        self.visit_return_type(&node.output);
    }

    fn visit_fn_arg(&mut self, node: &'ast syn::FnArg) {
        match node {
            syn::FnArg::Receiver(_binding_0) => {
                self.push_tokens(node.to_token_stream());
            }
            syn::FnArg::Typed(node) => {
                for it in &node.attrs {
                    self.visit_attribute(it);
                }
                self.visit_pat(&*node.pat);
                self.push_tokens(node.colon_token.to_token_stream());
                self.visit_type(&*node.ty);
            }
        }
    }

    fn visit_abi(&mut self, i: &'ast syn::Abi) {
        self.push_tokens(i.to_token_stream());
    }

    fn visit_angle_bracketed_generic_arguments(
        &mut self,
        i: &'ast syn::AngleBracketedGenericArguments
    ) {
        self.push_tokens(i.to_token_stream());
    }

    fn visit_arm(&mut self, _: &'ast syn::Arm) {
        unimplemented!("codegen::arm");
    }

    fn visit_assoc_const(&mut self, _: &'ast syn::AssocConst) {
        unimplemented!("codegen::assoc_const");
    }

    fn visit_assoc_type(&mut self, _: &'ast syn::AssocType) {
        unimplemented!("codegen::assoc_type");
    }

    fn visit_attr_style(&mut self, _: &'ast syn::AttrStyle) {
        unimplemented!("codegen::attr_style");
    }

    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        self.push_tokens(node.to_token_stream());
    }

    fn visit_bare_fn_arg(&mut self, _: &'ast syn::BareFnArg) {
        unimplemented!("codegen::bare_fn_arg");
    }

    fn visit_bare_variadic(&mut self, _: &'ast syn::BareVariadic) {
        unimplemented!("codegen::bare_variadic");
    }

    fn visit_bin_op(&mut self, op: &'ast syn::BinOp) {
        self.push_tokens(op.to_token_stream());
    }

    fn visit_bound_lifetimes(&mut self, _: &'ast syn::BoundLifetimes) {
        unimplemented!("codegen::bound_lifetimes");
    }

    fn visit_captured_param(&mut self, _: &'ast syn::CapturedParam) {
        unimplemented!("codegen::captured_param");
    }

    fn visit_const_param(&mut self, _: &'ast syn::ConstParam) {
        unimplemented!("codegen::const_param");
    }

    fn visit_constraint(&mut self, _: &'ast syn::Constraint) {
        unimplemented!("codegen::constraint");
    }

    fn visit_data(&mut self, _: &'ast syn::Data) {
        unimplemented!("codegen::data");
    }

    fn visit_data_enum(&mut self, _: &'ast syn::DataEnum) {
        unimplemented!("codegen::data_enum");
    }

    fn visit_data_struct(&mut self, _: &'ast syn::DataStruct) {
        unimplemented!("codegen::data_struct");
    }

    fn visit_data_union(&mut self, _: &'ast syn::DataUnion) {
        unimplemented!("codegen::data_union");
    }

    fn visit_derive_input(&mut self, _: &'ast syn::DeriveInput) {
        unimplemented!("codegen::derive_input");
    }

    fn visit_expr_array(&mut self, _: &'ast syn::ExprArray) {
        unimplemented!("codegen::expr_array");
    }

    fn visit_expr_async(&mut self, _: &'ast syn::ExprAsync) {
        unimplemented!("codegen::expr_async");
    }

    fn visit_expr_await(&mut self, _: &'ast syn::ExprAwait) {
        unimplemented!("codegen::expr_await");
    }

    fn visit_expr_break(&mut self, _: &'ast syn::ExprBreak) {
        unimplemented!("codegen::expr_break");
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.visit_expr(&*node.func);
        let mut current_token = self.current_tokens.drain(..).collect::<Vec<_>>();
        for el in syn::punctuated::Punctuated::pairs(&node.args) {
            let it = el.value();
            println!("visit_expr_call: {:#?}", it);
            self.visit_expr(it);
            if let Some(comma) = el.punct() {
                self.push_tokens(comma.to_token_stream());
            }
        }
        let arg_token = self.current_tokens.drain(..).collect::<TokenStream2>();
        current_token.push(quote::quote! { (#arg_token) });
        self.current_tokens = current_token;
    }

    fn visit_expr_cast(&mut self, _: &'ast syn::ExprCast) {
        unimplemented!("codegen::expr_cast");
    }

    fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {
        unimplemented!("codegen::expr_closure");
    }

    fn visit_expr_const(&mut self, _: &'ast syn::ExprConst) {
        unimplemented!("codegen::expr_const");
    }

    fn visit_expr_continue(&mut self, _: &'ast syn::ExprContinue) {
        unimplemented!("codegen::expr_continue");
    }

    fn visit_expr_field(&mut self, _: &'ast syn::ExprField) {
        unimplemented!("codegen::expr_field");
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        if let Some(it) = &node.label {
            self.visit_label(it);
        }
        self.push_tokens(node.for_token.to_token_stream());
        self.visit_pat(&*node.pat);
        self.push_tokens(node.in_token.to_token_stream());
        self.visit_expr(&*node.expr);
        self.visit_block(&node.body);
    }

    fn visit_expr_group(&mut self, _: &'ast syn::ExprGroup) {
        unimplemented!("codegen::expr_group");
    }

    fn visit_expr_if(&mut self, _: &'ast syn::ExprIf) {
        unimplemented!("codegen::expr_if");
    }

    fn visit_expr_index(&mut self, _: &'ast syn::ExprIndex) {
        unimplemented!("codegen::expr_index");
    }

    fn visit_expr_infer(&mut self, _: &'ast syn::ExprInfer) {
        unimplemented!("codegen::expr_infer");
    }

    fn visit_expr_let(&mut self, _: &'ast syn::ExprLet) {
        unimplemented!("codegen::expr_let");
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        self.push_tokens(i.to_token_stream());
    }

    fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {
        unimplemented!("codegen::expr_loop");
    }

    fn visit_expr_macro(&mut self, _: &'ast syn::ExprMacro) {
        unimplemented!("codegen::expr_macro");
    }

    fn visit_expr_match(&mut self, _: &'ast syn::ExprMatch) {
        unimplemented!("codegen::expr_match");
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.visit_expr(&*node.receiver);
        self.push_tokens(node.dot_token.to_token_stream());
        self.visit_ident(&node.method);
        if let Some(it) = &node.turbofish {
            self.visit_angle_bracketed_generic_arguments(it);
        }
        let mut current_token = self.current_tokens.drain(..).collect::<Vec<_>>();
        for el in syn::punctuated::Punctuated::pairs(&node.args) {
            let it = el.value();
            self.visit_expr(it);
            if let Some(comma) = el.punct() {
                self.push_tokens(comma.to_token_stream());
            }
        }
        let arg_token = self.current_tokens.drain(..).collect::<TokenStream2>();
        current_token.push(quote::quote! { (#arg_token) });
        self.current_tokens = current_token;
    }

    fn visit_expr_paren(&mut self, _: &'ast syn::ExprParen) {
        unimplemented!("codegen::expr_paren");
    }

    fn visit_expr_range(&mut self, range: &'ast syn::ExprRange) {
        self.push_tokens(range.to_token_stream());
    }

    fn visit_expr_raw_addr(&mut self, _: &'ast syn::ExprRawAddr) {
        unimplemented!("codegen::expr_raw_addr");
    }

    fn visit_expr_reference(&mut self, node: &'ast syn::ExprReference) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.push_tokens(node.and_token.to_token_stream());
        if let Some(it) = &node.mutability {
            self.push_tokens(it.to_token_stream());
        }
        self.visit_expr(&*node.expr);
    }

    fn visit_expr_repeat(&mut self, _: &'ast syn::ExprRepeat) {
        unimplemented!("codegen::expr_repeat");
    }

    fn visit_expr_return(&mut self, _: &'ast syn::ExprReturn) {
        unimplemented!("codegen::expr_return");
    }

    fn visit_expr_struct(&mut self, _: &'ast syn::ExprStruct) {
        unimplemented!("codegen::expr_struct");
    }

    fn visit_expr_try(&mut self, node: &'ast syn::ExprTry) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.visit_expr(&*node.expr);
        self.push_tokens(node.question_token.to_token_stream());
    }

    fn visit_expr_try_block(&mut self, _: &'ast syn::ExprTryBlock) {
        unimplemented!("codegen::expr_try_block");
    }

    fn visit_expr_tuple(&mut self, _: &'ast syn::ExprTuple) {
        unimplemented!("codegen::expr_tuple");
    }

    fn visit_expr_unary(&mut self, _: &'ast syn::ExprUnary) {
        unimplemented!("codegen::expr_unary");
    }

    fn visit_expr_unsafe(&mut self, _: &'ast syn::ExprUnsafe) {
        unimplemented!("codegen::expr_unsafe");
    }

    fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {
        unimplemented!("codegen::expr_while");
    }

    fn visit_expr_yield(&mut self, _: &'ast syn::ExprYield) {
        unimplemented!("codegen::expr_yield");
    }

    fn visit_field(&mut self, _: &'ast syn::Field) {
        unimplemented!("codegen::field");
    }

    fn visit_field_mutability(&mut self, _: &'ast syn::FieldMutability) {
        unimplemented!("codegen::field_mutability");
    }

    fn visit_field_pat(&mut self, _: &'ast syn::FieldPat) {
        unimplemented!("codegen::field_pat");
    }

    fn visit_field_value(&mut self, _: &'ast syn::FieldValue) {
        unimplemented!("codegen::field_value");
    }

    fn visit_fields(&mut self, _: &'ast syn::Fields) {
        unimplemented!("codegen::fields");
    }

    fn visit_fields_named(&mut self, _: &'ast syn::FieldsNamed) {
        unimplemented!("codegen::fields_named");
    }

    fn visit_fields_unnamed(&mut self, _: &'ast syn::FieldsUnnamed) {
        unimplemented!("codegen::fields_unnamed");
    }

    fn visit_file(&mut self, _: &'ast syn::File) {
        unimplemented!("codegen::file");
    }

    fn visit_foreign_item(&mut self, _: &'ast syn::ForeignItem) {
        unimplemented!("codegen::foreign_item");
    }

    fn visit_foreign_item_fn(&mut self, _: &'ast syn::ForeignItemFn) {
        unimplemented!("codegen::foreign_item_fn");
    }

    fn visit_foreign_item_macro(&mut self, _: &'ast syn::ForeignItemMacro) {
        unimplemented!("codegen::foreign_item_macro");
    }

    fn visit_foreign_item_static(&mut self, _: &'ast syn::ForeignItemStatic) {
        unimplemented!("codegen::foreign_item_static");
    }

    fn visit_foreign_item_type(&mut self, _: &'ast syn::ForeignItemType) {
        unimplemented!("codegen::foreign_item_type");
    }

    fn visit_generic_argument(&mut self, _: &'ast syn::GenericArgument) {
        unimplemented!("codegen::generic_argument");
    }

    fn visit_generic_param(&mut self, _: &'ast syn::GenericParam) {
        unimplemented!("codegen::generic_param");
    }

    fn visit_generics(&mut self, i: &'ast syn::Generics) {
        self.push_tokens(i.to_token_stream());
    }

    fn visit_impl_item(&mut self, _: &'ast syn::ImplItem) {
        unimplemented!("codegen::impl_item");
    }

    fn visit_impl_item_const(&mut self, _: &'ast syn::ImplItemConst) {
        unimplemented!("codegen::impl_item_const");
    }

    fn visit_impl_item_macro(&mut self, _: &'ast syn::ImplItemMacro) {
        unimplemented!("codegen::impl_item_macro");
    }

    fn visit_impl_item_type(&mut self, _: &'ast syn::ImplItemType) {
        unimplemented!("codegen::impl_item_type");
    }

    fn visit_impl_restriction(&mut self, _: &'ast syn::ImplRestriction) {
        unimplemented!("codegen::impl_restriction");
    }

    fn visit_index(&mut self, _: &'ast syn::Index) {
        unimplemented!("codegen::index");
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {
        unimplemented!("codegen::item");
    }

    fn visit_item_const(&mut self, _: &'ast syn::ItemConst) {
        unimplemented!("codegen::item_const");
    }

    fn visit_item_enum(&mut self, _: &'ast syn::ItemEnum) {
        unimplemented!("codegen::item_enum");
    }

    fn visit_item_extern_crate(&mut self, _: &'ast syn::ItemExternCrate) {
        unimplemented!("codegen::item_extern_crate");
    }

    fn visit_item_fn(&mut self, _: &'ast syn::ItemFn) {
        unimplemented!("codegen::item_fn");
    }

    fn visit_item_foreign_mod(&mut self, _: &'ast syn::ItemForeignMod) {
        unimplemented!("codegen::item_foreign_mod");
    }

    fn visit_item_impl(&mut self, _: &'ast syn::ItemImpl) {
        unimplemented!("codegen::item_impl");
    }

    fn visit_item_macro(&mut self, _: &'ast syn::ItemMacro) {
        unimplemented!("codegen::item_macro");
    }

    fn visit_item_mod(&mut self, _: &'ast syn::ItemMod) {
        unimplemented!("codegen::item_mod");
    }

    fn visit_item_static(&mut self, _: &'ast syn::ItemStatic) {
        unimplemented!("codegen::item_static");
    }

    fn visit_item_struct(&mut self, _: &'ast syn::ItemStruct) {
        unimplemented!("codegen::item_struct");
    }

    fn visit_item_trait(&mut self, _: &'ast syn::ItemTrait) {
        unimplemented!("codegen::item_trait");
    }

    fn visit_item_trait_alias(&mut self, _: &'ast syn::ItemTraitAlias) {
        unimplemented!("codegen::item_trait_alias");
    }

    fn visit_item_type(&mut self, _: &'ast syn::ItemType) {
        unimplemented!("codegen::item_type");
    }

    fn visit_item_union(&mut self, _: &'ast syn::ItemUnion) {
        unimplemented!("codegen::item_union");
    }

    fn visit_item_use(&mut self, _: &'ast syn::ItemUse) {
        unimplemented!("codegen::item_use");
    }

    fn visit_label(&mut self, _: &'ast syn::Label) {
        unimplemented!("codegen::label");
    }
    fn visit_lifetime(&mut self, _: &'ast syn::Lifetime) {
        unimplemented!("codegen::lifetime");
    }

    fn visit_lifetime_param(&mut self, _: &'ast syn::LifetimeParam) {
        unimplemented!("codegen::lifetime_param");
    }
    fn visit_lit(&mut self, _: &'ast syn::Lit) {
        unimplemented!("codegen::lit");
    }
    fn visit_lit_bool(&mut self, _: &'ast syn::LitBool) {
        unimplemented!("codegen::lit_bool");
    }
    fn visit_lit_byte(&mut self, _: &'ast syn::LitByte) {
        unimplemented!("codegen::lit_byte");
    }
    fn visit_lit_byte_str(&mut self, _: &'ast syn::LitByteStr) {
        unimplemented!("codegen::lit_byte_str");
    }
    fn visit_lit_cstr(&mut self, _: &'ast syn::LitCStr) {
        unimplemented!("codegen::lit_cstr");
    }
    fn visit_lit_char(&mut self, _: &'ast syn::LitChar) {
        unimplemented!("codegen::lit_char");
    }
    fn visit_lit_float(&mut self, _: &'ast syn::LitFloat) {
        unimplemented!("codegen::lit_float");
    }
    fn visit_lit_int(&mut self, _: &'ast syn::LitInt) {
        unimplemented!("codegen::lit_int");
    }
    fn visit_lit_str(&mut self, _: &'ast syn::LitStr) {
        unimplemented!("codegen::lit_str");
    }

    fn visit_local_init(&mut self, node: &'ast syn::LocalInit) {
        self.push_tokens(node.eq_token.to_token_stream());
        self.visit_expr(&*node.expr);
        if let Some(it) = &node.diverge {
            self.push_tokens(it.0.to_token_stream());
            self.visit_expr(&*it.1);
        }
    }

    fn visit_macro(&mut self, _: &'ast syn::Macro) {
        unimplemented!("codegen::macro");
    }

    fn visit_macro_delimiter(&mut self, _: &'ast syn::MacroDelimiter) {
        unimplemented!("codegen::macro_delimiter");
    }

    fn visit_member(&mut self, _: &'ast syn::Member) {
        unimplemented!("codegen::member");
    }

    fn visit_meta(&mut self, _: &'ast syn::Meta) {
        unimplemented!("codegen::meta");
    }

    fn visit_meta_list(&mut self, _: &'ast syn::MetaList) {
        unimplemented!("codegen::meta_list");
    }

    fn visit_meta_name_value(&mut self, _: &'ast syn::MetaNameValue) {
        unimplemented!("codegen::meta_name_value");
    }

    fn visit_parenthesized_generic_arguments(
        &mut self,
        _: &'ast syn::ParenthesizedGenericArguments
    ) {
        unimplemented!("codegen::parenthesized_generic_arguments");
    }

    fn visit_pat_or(&mut self, _: &'ast syn::PatOr) {
        unimplemented!("codegen::pat_or");
    }

    fn visit_pat_paren(&mut self, _: &'ast syn::PatParen) {
        unimplemented!("codegen::pat_paren");
    }

    fn visit_pat_reference(&mut self, _: &'ast syn::PatReference) {
        unimplemented!("codegen::pat_reference");
    }

    fn visit_pat_rest(&mut self, _: &'ast syn::PatRest) {
        unimplemented!("codegen::pat_rest");
    }

    fn visit_pat_slice(&mut self, _: &'ast syn::PatSlice) {
        unimplemented!("codegen::pat_slice");
    }

    fn visit_pat_struct(&mut self, _: &'ast syn::PatStruct) {
        unimplemented!("codegen::pat_struct");
    }

    fn visit_pat_tuple(&mut self, _: &'ast syn::PatTuple) {
        unimplemented!("codegen::pat_tuple");
    }

    fn visit_pat_tuple_struct(&mut self, _: &'ast syn::PatTupleStruct) {
        unimplemented!("codegen::pat_tuple_struct");
    }

    fn visit_pat_type(&mut self, node: &'ast syn::PatType) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.visit_pat(&*node.pat);
        self.push_tokens(node.colon_token.to_token_stream());
        self.push_tokens(node.ty.to_token_stream());
    }

    fn visit_pat_wild(&mut self, node: &'ast syn::PatWild) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        self.push_tokens(node.underscore_token.to_token_stream());
    }

    fn visit_path(&mut self, node: &'ast syn::Path) {
        if let Some(ident) = node.get_ident() {
            let name = ident.to_string();
            let current_name = self.ssa_ctx.borrow().current_name(&name);
            if let Some(ssa_name) = current_name {
                let ident = syn::Ident::new(&ssa_name, ident.span());
                self.push_tokens(ident.to_token_stream());
            } else {
                self.push_tokens(ident.to_token_stream());
            }
        } else {
            self.push_tokens(node.to_token_stream());
        }
    }

    fn visit_path_arguments(&mut self, _: &'ast syn::PathArguments) {
        unimplemented!("codegen::path_arguments");
    }

    fn visit_pointer_mutability(&mut self, _: &'ast syn::PointerMutability) {
        unimplemented!("codegen::pointer_mutability");
    }

    fn visit_precise_capture(&mut self, _: &'ast syn::PreciseCapture) {
        unimplemented!("codegen::precise_capture");
    }

    fn visit_predicate_lifetime(&mut self, _: &'ast syn::PredicateLifetime) {
        unimplemented!("codegen::predicate_lifetime");
    }

    fn visit_predicate_type(&mut self, _: &'ast syn::PredicateType) {
        unimplemented!("codegen::predicate_type");
    }

    fn visit_qself(&mut self, _: &'ast syn::QSelf) {
        unimplemented!("codegen::qself");
    }

    fn visit_range_limits(&mut self, _: &'ast syn::RangeLimits) {
        unimplemented!("codegen::range_limits");
    }

    fn visit_receiver(&mut self, _: &'ast syn::Receiver) {
        unimplemented!("codegen::receiver");
    }

    fn visit_return_type(&mut self, node: &'ast syn::ReturnType) {
        self.push_tokens(node.to_token_stream());
    }
    fn visit_span(&mut self, _: &proc_macro2::Span) {
        unimplemented!("codegen::span");
    }

    fn visit_static_mutability(&mut self, _: &'ast syn::StaticMutability) {
        unimplemented!("codegen::static_mutability");
    }

    fn visit_stmt_macro(&mut self, _: &'ast syn::StmtMacro) {
        unimplemented!("codegen::stmt_macro");
    }

    fn visit_trait_bound(&mut self, _: &'ast syn::TraitBound) {
        unimplemented!("codegen::trait_bound");
    }

    fn visit_trait_bound_modifier(&mut self, _: &'ast syn::TraitBoundModifier) {
        unimplemented!("codegen::trait_bound_modifier");
    }

    fn visit_trait_item(&mut self, _: &'ast syn::TraitItem) {
        unimplemented!("codegen::trait_item");
    }

    fn visit_trait_item_const(&mut self, _: &'ast syn::TraitItemConst) {
        unimplemented!("codegen::trait_item_const");
    }

    fn visit_trait_item_fn(&mut self, _: &'ast syn::TraitItemFn) {
        unimplemented!("codegen::trait_item_fn");
    }

    fn visit_trait_item_macro(&mut self, _: &'ast syn::TraitItemMacro) {
        unimplemented!("codegen::trait_item_macro");
    }

    fn visit_trait_item_type(&mut self, _: &'ast syn::TraitItemType) {
        unimplemented!("codegen::trait_item_type");
    }

    fn visit_type_array(&mut self, _: &'ast syn::TypeArray) {
        unimplemented!("codegen::type_array");
    }

    fn visit_type_bare_fn(&mut self, _: &'ast syn::TypeBareFn) {
        unimplemented!("codegen::type_bare_fn");
    }

    fn visit_type_group(&mut self, _: &'ast syn::TypeGroup) {
        unimplemented!("codegen::type_group");
    }

    fn visit_type_impl_trait(&mut self, _: &'ast syn::TypeImplTrait) {
        unimplemented!("codegen::type_impl_trait");
    }

    fn visit_type_infer(&mut self, _: &'ast syn::TypeInfer) {
        unimplemented!("codegen::type_infer");
    }

    fn visit_type_macro(&mut self, _: &'ast syn::TypeMacro) {
        unimplemented!("codegen::type_macro");
    }

    fn visit_type_never(&mut self, _: &'ast syn::TypeNever) {
        unimplemented!("codegen::type_never");
    }

    fn visit_type_param(&mut self, _: &'ast syn::TypeParam) {
        unimplemented!("codegen::type_param");
    }

    fn visit_type_param_bound(&mut self, _: &'ast syn::TypeParamBound) {
        unimplemented!("codegen::type_param_bound");
    }

    fn visit_type_paren(&mut self, _: &'ast syn::TypeParen) {
        unimplemented!("codegen::type_paren");
    }

    fn visit_type_path(&mut self, _: &'ast syn::TypePath) {
        unimplemented!("codegen::type_path");
    }

    fn visit_type_ptr(&mut self, _: &'ast syn::TypePtr) {
        unimplemented!("codegen::type_ptr");
    }

    fn visit_type_reference(&mut self, _: &'ast syn::TypeReference) {
        unimplemented!("codegen::type_reference");
    }

    fn visit_type_slice(&mut self, _: &'ast syn::TypeSlice) {
        unimplemented!("codegen::type_slice");
    }

    fn visit_type_trait_object(&mut self, _: &'ast syn::TypeTraitObject) {
        unimplemented!("codegen::type_trait_object");
    }

    fn visit_type_tuple(&mut self, _: &'ast syn::TypeTuple) {
        unimplemented!("codegen::type_tuple");
    }

    fn visit_un_op(&mut self, _: &'ast syn::UnOp) {
        unimplemented!("codegen::un_op");
    }

    fn visit_use_glob(&mut self, _: &'ast syn::UseGlob) {
        unimplemented!("codegen::use_glob");
    }

    fn visit_use_group(&mut self, _: &'ast syn::UseGroup) {
        unimplemented!("codegen::use_group");
    }

    fn visit_use_name(&mut self, _: &'ast syn::UseName) {
        unimplemented!("codegen::use_name");
    }

    fn visit_use_path(&mut self, _: &'ast syn::UsePath) {
        unimplemented!("codegen::use_path");
    }

    fn visit_use_rename(&mut self, _: &'ast syn::UseRename) {
        unimplemented!("codegen::use_rename");
    }

    fn visit_use_tree(&mut self, _: &'ast syn::UseTree) {
        unimplemented!("codegen::use_tree");
    }

    fn visit_variadic(&mut self, _: &'ast syn::Variadic) {
        unimplemented!("codegen::variadic");
    }

    fn visit_variant(&mut self, _: &'ast syn::Variant) {
        unimplemented!("codegen::variant");
    }

    fn visit_vis_restricted(&mut self, _: &'ast syn::VisRestricted) {
        unimplemented!("codegen::vis_restricted");
    }

    fn visit_visibility(&mut self, node: &'ast syn::Visibility) {
        self.push_tokens(node.to_token_stream());
    }

    fn visit_where_clause(&mut self, _: &'ast syn::WhereClause) {
        unimplemented!("codegen::where_clause");
    }

    fn visit_where_predicate(&mut self, _: &'ast syn::WherePredicate) {
        unimplemented!("codegen::where_predicate");
    }

    fn visit_expr_block(&mut self, _: &'ast syn::ExprBlock) {
        unimplemented!("codegen::expr_block");
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        let mut new_codegen = _Codegen {
            fused_codes: self.fused_codes,
            to_remove: self.to_remove,
            current_tokens: Vec::new(),
            ssa_ctx: RCMut::new(SSAContext::new()),
            _visitor: if let Some(visitor) = self._visitor {
                visitor.next_visitor.as_ref()
            } else {
                None
            },
            next_codegen: None,
            pat_ident_need_remove: false,
            pat_ident_is_ret: false,
        };
        new_codegen.ssa_ctx.borrow_mut().prev_ssa_ctx = Some(self.ssa_ctx.clone());
        syn::visit::visit_block(&mut new_codegen, block);
        let code = new_codegen.current_tokens.drain(..).collect::<TokenStream2>();
        self.current_tokens.push(quote::quote! {{#code}});
        self.next_codegen = Some(Box::new(new_codegen));
    }

    fn visit_local(&mut self, node: &'ast syn::Local) {
        for it in &node.attrs {
            self.visit_attribute(it);
        }
        let mut current_tokens = self.current_tokens.drain(..).collect::<Vec<_>>();
        self.visit_pat(&node.pat);
        if !self.pat_ident_need_remove && !self.pat_ident_is_ret {
            let pat = self.current_tokens.drain(..).collect::<TokenStream2>();
            current_tokens.push(node.let_token.to_token_stream());
            current_tokens.push(pat);
            if let Some(it) = &node.init {
                self.visit_local_init(it);
            }
            self.push_tokens(node.semi_token.to_token_stream());
        }
        if self.pat_ident_is_ret {
            current_tokens.push(node.let_token.to_token_stream());
        }
        current_tokens.push(self.current_tokens.drain(..).collect::<TokenStream2>());
        self.current_tokens = current_tokens;
    }

    fn visit_ident(&mut self, i: &'ast proc_macro2::Ident) {
        self.push_tokens(i.to_token_stream());
    }

    fn visit_pat_ident(&mut self, node: &'ast syn::PatIdent) {
        let name = node.ident.to_string();
        let ssa_name = proc_macro2::Ident::new(
            &self.ssa_ctx.borrow_mut().fresh_name(&name),
            node.ident.span()
        );
        if !self.to_remove.iter().any(|set| set.contains(&ssa_name)) {
            if self.fused_codes.contains_key(&ssa_name) {
                self.push_tokens(self.fused_codes[&ssa_name].clone());
                self.pat_ident_is_ret = true;
            } else {
                self.push_tokens(ssa_name.to_token_stream());
            }
            self.pat_ident_need_remove = false;
        } else {
            self.pat_ident_need_remove = true;
        }
    }

    fn visit_type(&mut self, node: &'ast syn::Type) {
        self.push_tokens(node.to_token_stream());
    }
}
