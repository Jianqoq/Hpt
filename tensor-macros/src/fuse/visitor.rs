use quote::ToTokens;
use syn::{ spanned::Spanned, visit::* };

use super::node::{ Binary, Node, Unary };

pub(crate) struct Visitor<'ast> {
    pub(crate) nodes: Vec<Node<'ast>>,
    pub(crate) var_cnt: usize,
    pub(crate) current_var: proc_macro2::Ident,
    pub(crate) code: proc_macro2::TokenStream,
    pub(crate) current_assignment: Option<proc_macro2::Ident>,
}

impl<'ast> Visitor<'ast> {
    pub(crate) fn new() -> Self {
        use proc_macro2::Span;
        Self {
            nodes: vec![],
            var_cnt: 0,
            current_var: proc_macro2::Ident::new("__out0", Span::call_site()),
            code: proc_macro2::TokenStream::new(),
            current_assignment: None,
        }
    }
}

impl<'ast> Visit<'ast> for Visitor<'ast> {
    fn visit_abi(&mut self, i: &'ast syn::Abi) {
        visit_abi(self, i);
    }

    fn visit_angle_bracketed_generic_arguments(
        &mut self,
        i: &'ast syn::AngleBracketedGenericArguments
    ) {
        visit_angle_bracketed_generic_arguments(self, i);
    }
    fn visit_arm(&mut self, i: &'ast syn::Arm) {
        visit_arm(self, i);
    }
    fn visit_assoc_const(&mut self, i: &'ast syn::AssocConst) {
        visit_assoc_const(self, i);
    }

    fn visit_assoc_type(&mut self, i: &'ast syn::AssocType) {
        visit_assoc_type(self, i);
    }

    fn visit_attr_style(&mut self, i: &'ast syn::AttrStyle) {
        visit_attr_style(self, i);
    }

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        visit_attribute(self, i);
    }

    fn visit_bare_fn_arg(&mut self, i: &'ast syn::BareFnArg) {
        visit_bare_fn_arg(self, i);
    }

    fn visit_bare_variadic(&mut self, i: &'ast syn::BareVariadic) {
        visit_bare_variadic(self, i);
    }

    fn visit_bin_op(&mut self, i: &'ast syn::BinOp) {
        visit_bin_op(self, i);
    }

    fn visit_block(&mut self, i: &'ast syn::Block) {
        visit_block(self, i);
    }

    fn visit_bound_lifetimes(&mut self, i: &'ast syn::BoundLifetimes) {
        visit_bound_lifetimes(self, i);
    }

    fn visit_captured_param(&mut self, i: &'ast syn::CapturedParam) {
        visit_captured_param(self, i);
    }

    fn visit_const_param(&mut self, i: &'ast syn::ConstParam) {
        visit_const_param(self, i);
    }

    fn visit_constraint(&mut self, i: &'ast syn::Constraint) {
        visit_constraint(self, i);
    }

    fn visit_data(&mut self, i: &'ast syn::Data) {
        visit_data(self, i);
    }

    fn visit_data_enum(&mut self, i: &'ast syn::DataEnum) {
        visit_data_enum(self, i);
    }

    fn visit_data_struct(&mut self, i: &'ast syn::DataStruct) {
        visit_data_struct(self, i);
    }

    fn visit_data_union(&mut self, i: &'ast syn::DataUnion) {
        visit_data_union(self, i);
    }

    fn visit_derive_input(&mut self, i: &'ast syn::DeriveInput) {
        visit_derive_input(self, i);
    }

    fn visit_expr(&mut self, i: &'ast syn::Expr) {
        visit_expr(self, i);
    }

    fn visit_expr_array(&mut self, i: &'ast syn::ExprArray) {
        visit_expr_array(self, i);
    }

    fn visit_expr_assign(&mut self, i: &'ast syn::ExprAssign) {
        visit_expr_assign(self, i);
    }

    fn visit_expr_async(&mut self, i: &'ast syn::ExprAsync) {
        visit_expr_async(self, i);
    }

    fn visit_expr_await(&mut self, i: &'ast syn::ExprAwait) {
        visit_expr_await(self, i);
    }

    fn visit_expr_block(&mut self, i: &'ast syn::ExprBlock) {
        visit_expr_block(self, i);
    }

    fn visit_expr_break(&mut self, i: &'ast syn::ExprBreak) {
        visit_expr_break(self, i);
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        visit_expr_call(self, i);
    }

    fn visit_expr_cast(&mut self, i: &'ast syn::ExprCast) {
        visit_expr_cast(self, i);
    }

    fn visit_expr_closure(&mut self, i: &'ast syn::ExprClosure) {
        visit_expr_closure(self, i);
    }

    fn visit_expr_const(&mut self, i: &'ast syn::ExprConst) {
        visit_expr_const(self, i);
    }

    fn visit_expr_continue(&mut self, i: &'ast syn::ExprContinue) {
        visit_expr_continue(self, i);
    }

    fn visit_expr_field(&mut self, i: &'ast syn::ExprField) {
        visit_expr_field(self, i);
    }

    fn visit_expr_for_loop(&mut self, i: &'ast syn::ExprForLoop) {
        visit_expr_for_loop(self, i);
    }

    fn visit_expr_group(&mut self, i: &'ast syn::ExprGroup) {
        visit_expr_group(self, i);
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        visit_expr_if(self, i);
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        visit_expr_index(self, i);
    }

    fn visit_expr_infer(&mut self, i: &'ast syn::ExprInfer) {
        visit_expr_infer(self, i);
    }

    fn visit_expr_let(&mut self, i: &'ast syn::ExprLet) {
        visit_expr_let(self, i);
    }

    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        visit_expr_lit(self, i);
    }

    fn visit_expr_loop(&mut self, i: &'ast syn::ExprLoop) {
        visit_expr_loop(self, i);
    }

    fn visit_expr_macro(&mut self, i: &'ast syn::ExprMacro) {
        visit_expr_macro(self, i);
    }

    fn visit_expr_match(&mut self, i: &'ast syn::ExprMatch) {
        visit_expr_match(self, i);
    }

    fn visit_expr_paren(&mut self, i: &'ast syn::ExprParen) {
        visit_expr_paren(self, i);
    }

    fn visit_expr_range(&mut self, i: &'ast syn::ExprRange) {
        visit_expr_range(self, i);
    }

    fn visit_expr_raw_addr(&mut self, i: &'ast syn::ExprRawAddr) {
        visit_expr_raw_addr(self, i);
    }
    fn visit_expr_repeat(&mut self, i: &'ast syn::ExprRepeat) {
        visit_expr_repeat(self, i);
    }

    fn visit_expr_return(&mut self, i: &'ast syn::ExprReturn) {
        visit_expr_return(self, i);
    }

    fn visit_expr_struct(&mut self, i: &'ast syn::ExprStruct) {
        visit_expr_struct(self, i);
    }

    fn visit_expr_try(&mut self, i: &'ast syn::ExprTry) {
        visit_expr_try(self, i);
    }

    fn visit_expr_try_block(&mut self, i: &'ast syn::ExprTryBlock) {
        visit_expr_try_block(self, i);
    }

    fn visit_expr_tuple(&mut self, i: &'ast syn::ExprTuple) {
        visit_expr_tuple(self, i);
    }

    fn visit_expr_unary(&mut self, i: &'ast syn::ExprUnary) {
        visit_expr_unary(self, i);
    }

    fn visit_expr_unsafe(&mut self, i: &'ast syn::ExprUnsafe) {
        visit_expr_unsafe(self, i);
    }

    fn visit_expr_while(&mut self, i: &'ast syn::ExprWhile) {
        visit_expr_while(self, i);
    }

    fn visit_expr_yield(&mut self, i: &'ast syn::ExprYield) {
        visit_expr_yield(self, i);
    }

    fn visit_field(&mut self, i: &'ast syn::Field) {
        visit_field(self, i);
    }

    fn visit_field_mutability(&mut self, i: &'ast syn::FieldMutability) {
        visit_field_mutability(self, i);
    }

    fn visit_field_pat(&mut self, i: &'ast syn::FieldPat) {
        visit_field_pat(self, i);
    }

    fn visit_field_value(&mut self, i: &'ast syn::FieldValue) {
        visit_field_value(self, i);
    }

    fn visit_fields(&mut self, i: &'ast syn::Fields) {
        visit_fields(self, i);
    }

    fn visit_fields_named(&mut self, i: &'ast syn::FieldsNamed) {
        visit_fields_named(self, i);
    }

    fn visit_fields_unnamed(&mut self, i: &'ast syn::FieldsUnnamed) {
        visit_fields_unnamed(self, i);
    }

    fn visit_file(&mut self, i: &'ast syn::File) {
        visit_file(self, i);
    }

    fn visit_fn_arg(&mut self, i: &'ast syn::FnArg) {
        visit_fn_arg(self, i);
    }

    fn visit_foreign_item(&mut self, i: &'ast syn::ForeignItem) {
        visit_foreign_item(self, i);
    }

    fn visit_foreign_item_fn(&mut self, i: &'ast syn::ForeignItemFn) {
        visit_foreign_item_fn(self, i);
    }

    fn visit_foreign_item_macro(&mut self, i: &'ast syn::ForeignItemMacro) {
        visit_foreign_item_macro(self, i);
    }

    fn visit_foreign_item_static(&mut self, i: &'ast syn::ForeignItemStatic) {
        visit_foreign_item_static(self, i);
    }

    fn visit_foreign_item_type(&mut self, i: &'ast syn::ForeignItemType) {
        visit_foreign_item_type(self, i);
    }

    fn visit_generic_argument(&mut self, i: &'ast syn::GenericArgument) {
        visit_generic_argument(self, i);
    }

    fn visit_generic_param(&mut self, i: &'ast syn::GenericParam) {
        visit_generic_param(self, i);
    }

    fn visit_generics(&mut self, i: &'ast syn::Generics) {
        visit_generics(self, i);
    }

    fn visit_impl_item(&mut self, i: &'ast syn::ImplItem) {
        visit_impl_item(self, i);
    }

    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        visit_impl_item_const(self, i);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        visit_impl_item_fn(self, i);
    }

    fn visit_impl_item_macro(&mut self, i: &'ast syn::ImplItemMacro) {
        visit_impl_item_macro(self, i);
    }

    fn visit_impl_item_type(&mut self, i: &'ast syn::ImplItemType) {
        visit_impl_item_type(self, i);
    }

    fn visit_impl_restriction(&mut self, i: &'ast syn::ImplRestriction) {
        visit_impl_restriction(self, i);
    }

    fn visit_index(&mut self, i: &'ast syn::Index) {
        visit_index(self, i);
    }

    fn visit_item(&mut self, i: &'ast syn::Item) {
        visit_item(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        visit_item_const(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        visit_item_enum(self, i);
    }

    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        visit_item_extern_crate(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        visit_item_fn(self, i);
    }

    fn visit_item_foreign_mod(&mut self, i: &'ast syn::ItemForeignMod) {
        visit_item_foreign_mod(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        visit_item_impl(self, i);
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        visit_item_macro(self, i);
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        visit_item_mod(self, i);
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        visit_item_static(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        visit_item_struct(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        visit_item_trait(self, i);
    }

    fn visit_item_trait_alias(&mut self, i: &'ast syn::ItemTraitAlias) {
        visit_item_trait_alias(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        visit_item_type(self, i);
    }

    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        visit_item_union(self, i);
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        visit_item_use(self, i);
    }

    fn visit_label(&mut self, i: &'ast syn::Label) {
        visit_label(self, i);
    }
    fn visit_lifetime(&mut self, i: &'ast syn::Lifetime) {
        visit_lifetime(self, i);
    }

    fn visit_lifetime_param(&mut self, i: &'ast syn::LifetimeParam) {
        visit_lifetime_param(self, i);
    }
    fn visit_lit(&mut self, i: &'ast syn::Lit) {
        visit_lit(self, i);
    }
    fn visit_lit_bool(&mut self, i: &'ast syn::LitBool) {
        visit_lit_bool(self, i);
    }
    fn visit_lit_byte(&mut self, i: &'ast syn::LitByte) {
        visit_lit_byte(self, i);
    }
    fn visit_lit_byte_str(&mut self, i: &'ast syn::LitByteStr) {
        visit_lit_byte_str(self, i);
    }
    fn visit_lit_cstr(&mut self, i: &'ast syn::LitCStr) {
        visit_lit_cstr(self, i);
    }
    fn visit_lit_char(&mut self, i: &'ast syn::LitChar) {
        visit_lit_char(self, i);
    }
    fn visit_lit_float(&mut self, i: &'ast syn::LitFloat) {
        visit_lit_float(self, i);
    }
    fn visit_lit_int(&mut self, i: &'ast syn::LitInt) {
        visit_lit_int(self, i);
    }
    fn visit_lit_str(&mut self, i: &'ast syn::LitStr) {
        visit_lit_str(self, i);
    }
    fn visit_local_init(&mut self, i: &'ast syn::LocalInit) {
        visit_local_init(self, i);
    }
    fn visit_macro(&mut self, i: &'ast syn::Macro) {
        visit_macro(self, i);
    }
    fn visit_macro_delimiter(&mut self, i: &'ast syn::MacroDelimiter) {
        visit_macro_delimiter(self, i);
    }
    fn visit_member(&mut self, i: &'ast syn::Member) {
        visit_member(self, i);
    }
    fn visit_meta(&mut self, i: &'ast syn::Meta) {
        visit_meta(self, i);
    }
    fn visit_meta_list(&mut self, i: &'ast syn::MetaList) {
        visit_meta_list(self, i);
    }
    fn visit_meta_name_value(&mut self, i: &'ast syn::MetaNameValue) {
        visit_meta_name_value(self, i);
    }
    fn visit_parenthesized_generic_arguments(
        &mut self,
        i: &'ast syn::ParenthesizedGenericArguments
    ) {
        visit_parenthesized_generic_arguments(self, i);
    }
    fn visit_pat(&mut self, i: &'ast syn::Pat) {
        visit_pat(self, i);
    }
    fn visit_pat_ident(&mut self, i: &'ast syn::PatIdent) {
        visit_pat_ident(self, i);
    }
    fn visit_pat_or(&mut self, i: &'ast syn::PatOr) {
        visit_pat_or(self, i);
    }
    fn visit_pat_paren(&mut self, i: &'ast syn::PatParen) {
        visit_pat_paren(self, i);
    }
    fn visit_pat_reference(&mut self, i: &'ast syn::PatReference) {
        visit_pat_reference(self, i);
    }
    fn visit_pat_rest(&mut self, i: &'ast syn::PatRest) {
        visit_pat_rest(self, i);
    }
    fn visit_pat_slice(&mut self, i: &'ast syn::PatSlice) {
        visit_pat_slice(self, i);
    }
    fn visit_pat_struct(&mut self, i: &'ast syn::PatStruct) {
        visit_pat_struct(self, i);
    }
    fn visit_pat_tuple(&mut self, i: &'ast syn::PatTuple) {
        visit_pat_tuple(self, i);
    }
    fn visit_pat_tuple_struct(&mut self, i: &'ast syn::PatTupleStruct) {
        visit_pat_tuple_struct(self, i);
    }
    fn visit_pat_type(&mut self, i: &'ast syn::PatType) {
        visit_pat_type(self, i);
    }
    fn visit_pat_wild(&mut self, i: &'ast syn::PatWild) {
        visit_pat_wild(self, i);
    }
    fn visit_path(&mut self, i: &'ast syn::Path) {
        visit_path(self, i);
    }

    fn visit_path_arguments(&mut self, i: &'ast syn::PathArguments) {
        visit_path_arguments(self, i);
    }
    fn visit_path_segment(&mut self, i: &'ast syn::PathSegment) {
        visit_path_segment(self, i);
    }
    fn visit_pointer_mutability(&mut self, i: &'ast syn::PointerMutability) {
        visit_pointer_mutability(self, i);
    }
    fn visit_precise_capture(&mut self, i: &'ast syn::PreciseCapture) {
        visit_precise_capture(self, i);
    }
    fn visit_predicate_lifetime(&mut self, i: &'ast syn::PredicateLifetime) {
        visit_predicate_lifetime(self, i);
    }
    fn visit_predicate_type(&mut self, i: &'ast syn::PredicateType) {
        visit_predicate_type(self, i);
    }
    fn visit_qself(&mut self, i: &'ast syn::QSelf) {
        visit_qself(self, i);
    }
    fn visit_range_limits(&mut self, i: &'ast syn::RangeLimits) {
        visit_range_limits(self, i);
    }
    fn visit_receiver(&mut self, i: &'ast syn::Receiver) {
        visit_receiver(self, i);
    }
    fn visit_return_type(&mut self, i: &'ast syn::ReturnType) {
        visit_return_type(self, i);
    }
    fn visit_signature(&mut self, i: &'ast syn::Signature) {
        visit_signature(self, i);
    }
    fn visit_span(&mut self, i: &proc_macro2::Span) {
        visit_span(self, i);
    }
    fn visit_static_mutability(&mut self, i: &'ast syn::StaticMutability) {
        visit_static_mutability(self, i);
    }
    fn visit_stmt(&mut self, i: &'ast syn::Stmt) {
        visit_stmt(self, i);
    }
    fn visit_stmt_macro(&mut self, i: &'ast syn::StmtMacro) {
        visit_stmt_macro(self, i);
    }
    fn visit_trait_bound(&mut self, i: &'ast syn::TraitBound) {
        visit_trait_bound(self, i);
    }
    fn visit_trait_bound_modifier(&mut self, i: &'ast syn::TraitBoundModifier) {
        visit_trait_bound_modifier(self, i);
    }
    fn visit_trait_item(&mut self, i: &'ast syn::TraitItem) {
        visit_trait_item(self, i);
    }
    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        visit_trait_item_const(self, i);
    }
    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        visit_trait_item_fn(self, i);
    }
    fn visit_trait_item_macro(&mut self, i: &'ast syn::TraitItemMacro) {
        visit_trait_item_macro(self, i);
    }
    fn visit_trait_item_type(&mut self, i: &'ast syn::TraitItemType) {
        visit_trait_item_type(self, i);
    }
    fn visit_type(&mut self, i: &'ast syn::Type) {
        visit_type(self, i);
    }
    fn visit_type_array(&mut self, i: &'ast syn::TypeArray) {
        visit_type_array(self, i);
    }
    fn visit_type_bare_fn(&mut self, i: &'ast syn::TypeBareFn) {
        visit_type_bare_fn(self, i);
    }
    fn visit_type_group(&mut self, i: &'ast syn::TypeGroup) {
        visit_type_group(self, i);
    }
    fn visit_type_impl_trait(&mut self, i: &'ast syn::TypeImplTrait) {
        visit_type_impl_trait(self, i);
    }
    fn visit_type_infer(&mut self, i: &'ast syn::TypeInfer) {
        visit_type_infer(self, i);
    }
    fn visit_type_macro(&mut self, i: &'ast syn::TypeMacro) {
        visit_type_macro(self, i);
    }
    fn visit_type_never(&mut self, i: &'ast syn::TypeNever) {
        visit_type_never(self, i);
    }
    fn visit_type_param(&mut self, i: &'ast syn::TypeParam) {
        visit_type_param(self, i);
    }
    fn visit_type_param_bound(&mut self, i: &'ast syn::TypeParamBound) {
        visit_type_param_bound(self, i);
    }
    fn visit_type_paren(&mut self, i: &'ast syn::TypeParen) {
        visit_type_paren(self, i);
    }
    fn visit_type_path(&mut self, i: &'ast syn::TypePath) {
        visit_type_path(self, i);
    }
    fn visit_type_ptr(&mut self, i: &'ast syn::TypePtr) {
        visit_type_ptr(self, i);
    }
    fn visit_type_reference(&mut self, i: &'ast syn::TypeReference) {
        visit_type_reference(self, i);
    }
    fn visit_type_slice(&mut self, i: &'ast syn::TypeSlice) {
        visit_type_slice(self, i);
    }
    fn visit_type_trait_object(&mut self, i: &'ast syn::TypeTraitObject) {
        visit_type_trait_object(self, i);
    }
    fn visit_type_tuple(&mut self, i: &'ast syn::TypeTuple) {
        visit_type_tuple(self, i);
    }
    fn visit_un_op(&mut self, i: &'ast syn::UnOp) {
        visit_un_op(self, i);
    }
    fn visit_use_glob(&mut self, i: &'ast syn::UseGlob) {
        visit_use_glob(self, i);
    }
    fn visit_use_group(&mut self, i: &'ast syn::UseGroup) {
        visit_use_group(self, i);
    }
    fn visit_use_name(&mut self, i: &'ast syn::UseName) {
        visit_use_name(self, i);
    }
    fn visit_use_path(&mut self, i: &'ast syn::UsePath) {
        visit_use_path(self, i);
    }
    fn visit_use_rename(&mut self, i: &'ast syn::UseRename) {
        visit_use_rename(self, i);
    }
    fn visit_use_tree(&mut self, i: &'ast syn::UseTree) {
        visit_use_tree(self, i);
    }
    fn visit_variadic(&mut self, i: &'ast syn::Variadic) {
        visit_variadic(self, i);
    }
    fn visit_variant(&mut self, i: &'ast syn::Variant) {
        visit_variant(self, i);
    }
    fn visit_vis_restricted(&mut self, i: &'ast syn::VisRestricted) {
        visit_vis_restricted(self, i);
    }
    fn visit_visibility(&mut self, i: &'ast syn::Visibility) {
        visit_visibility(self, i);
    }
    fn visit_where_clause(&mut self, i: &'ast syn::WhereClause) {
        visit_where_clause(self, i);
    }
    fn visit_where_predicate(&mut self, i: &'ast syn::WherePredicate) {
        visit_where_predicate(self, i);
    }
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
        self.visit_expr(&i.left);
        let left_var = self.current_var.clone();
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
