use std::collections::{ HashMap, HashSet };

use quote::ToTokens;
use syn::{ visit::*, visit_mut::VisitMut };
use crate::TokenStream2;

use super::{ ssa::SSAContext, visitor::_Visitor };

pub(crate) struct Codegen<'ast> {
    pub(crate) _codegen: _Codegen<'ast>,
}

impl<'ast> Codegen<'ast> {
    pub(crate) fn get_code(&mut self) -> TokenStream2 {
        if let Some(next_codegen) = &mut self._codegen.next_codegen {
            let mut token_stream = TokenStream2::new();
            token_stream.extend(next_codegen.get_code());
            token_stream
        } else {
            self._codegen.get_code()
        }
    }
}

impl<'ast> Visit<'ast> for Codegen<'ast> {
    fn visit_block(&mut self, i: &'ast syn::Block) {
        visit_block(&mut self._codegen, i);
    }
}

pub(crate) struct _Codegen<'ast> {
    pub(crate) fused_codes: &'ast HashMap<syn::Ident, TokenStream2>,
    pub(crate) to_remove: &'ast Vec<HashSet<syn::Ident>>,
    pub(crate) current_tokens: Vec<TokenStream2>,
    pub(crate) ssa_ctx: SSAContext,
    pub(crate) _visitor: Option<&'ast Box<_Visitor<'ast>>>,
    pub(crate) next_codegen: Option<Box<_Codegen<'ast>>>,
}

impl<'ast> _Codegen<'ast> {
    fn push_tokens(&mut self, tokens: TokenStream2) {
        self.current_tokens.push(tokens);
    }

    pub(crate) fn get_code(&mut self) -> TokenStream2 {
        self.current_tokens.drain(..).collect::<TokenStream2>()
    }

    fn convert_stmt_to_ssa(&self, stmt: &mut syn::Stmt) -> TokenStream2 {
        struct SSAReplacer<'a> {
            ssa_ctx: &'a SSAContext,
        }
        // println!("stmt: {:#?}", stmt);

        impl<'a> VisitMut for SSAReplacer<'a> {
            fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
                match stmt {
                    syn::Stmt::Local(local) => {
                        match &mut local.pat {
                            syn::Pat::Const(_) => todo!("codegen::const"),
                            syn::Pat::Ident(pat_ident) => {
                                let name = pat_ident.ident.to_string();
                                let ssa_name = self.ssa_ctx.current_name(&name).unwrap();
                                pat_ident.ident = syn::Ident::new(
                                    &ssa_name,
                                    pat_ident.ident.span()
                                );
                            }
                            syn::Pat::Lit(_) => todo!("codegen::lit"),
                            syn::Pat::Macro(_) => todo!("codegen::macro"),
                            syn::Pat::Or(_) => todo!("codegen::or"),
                            syn::Pat::Paren(_) => todo!("codegen::paren"),
                            syn::Pat::Path(_) => todo!("codegen::path"),
                            syn::Pat::Range(_) => todo!("codegen::range"),
                            syn::Pat::Reference(_) => todo!("codegen::reference"),
                            syn::Pat::Rest(_) => todo!("codegen::rest"),
                            syn::Pat::Slice(_) => todo!("codegen::slice"),
                            syn::Pat::Struct(_) => todo!("codegen::struct"),
                            syn::Pat::Tuple(_) => todo!("codegen::tuple"),
                            syn::Pat::TupleStruct(_) => todo!("codegen::tuple_struct"),
                            syn::Pat::Type(_) => {}
                            syn::Pat::Verbatim(_) => todo!("codegen::verbatim"),
                            syn::Pat::Wild(_) => todo!("codegen::wild"),
                            _ => {}
                        }
                    }
                    _ => {
                        syn::visit_mut::visit_stmt_mut(self, stmt);
                    }
                }
            }
            fn visit_expr_path_mut(&mut self, expr: &mut syn::ExprPath) {
                if expr.path.segments.len() == 1 {
                    let ident = &expr.path.segments[0].ident;
                    if let Some(current_name) = self.ssa_ctx.current_name(&ident.to_string()) {
                        // 替换为当前的 SSA 名称
                        expr.path.segments[0].ident = syn::Ident::new(current_name, ident.span());
                    }
                }
                syn::visit_mut::visit_expr_path_mut(self, expr);
            }

            fn visit_expr_method_call_mut(&mut self, expr: &mut syn::ExprMethodCall) {
                self.visit_expr_mut(&mut *expr.receiver);
                for mut el in syn::punctuated::Punctuated::pairs_mut(&mut expr.args) {
                    let it = el.value_mut();
                    // println!("expr: {}", it.to_token_stream().to_string());
                    if let Some(current_name) = self.ssa_ctx.current_name_expr(it) {
                        if let syn::Expr::Path(path) = it {
                            path.path.segments[0].ident = syn::Ident::new(
                                current_name,
                                path.path.segments[0].ident.span()
                            );
                        }
                    } else {
                        self.visit_expr_mut(it);
                    }
                }
            }

            fn visit_expr_tuple_mut(&mut self, tuple: &mut syn::ExprTuple) {
                for mut el in syn::punctuated::Punctuated::pairs_mut(&mut tuple.elems) {
                    let it = el.value_mut();
                    if let Some(current_name) = self.ssa_ctx.current_name_expr(it) {
                        if let syn::Expr::Path(path) = it {
                            path.path.segments[0].ident = syn::Ident::new(
                                current_name,
                                path.path.segments[0].ident.span()
                            );
                        }
                    } else {
                        self.visit_expr_mut(it);
                    }
                }
            }

            fn visit_expr_call_mut(&mut self, expr: &mut syn::ExprCall) {
                println!("expr: {:#?}", expr);
                for mut el in syn::punctuated::Punctuated::pairs_mut(&mut expr.args) {
                    let it = el.value_mut();
                    println!("expr: {}", it.to_token_stream().to_string());
                    if let Some(current_name) = self.ssa_ctx.current_name_expr(it) {
                        if let syn::Expr::Path(path) = it {
                            path.path.segments[0].ident = syn::Ident::new(
                                current_name,
                                path.path.segments[0].ident.span()
                            );
                        }
                    } else {
                        self.visit_expr_mut(it);
                    }
                }
            }

            fn visit_type_mut(&mut self, _: &mut syn::Type) {}
        }

        let mut replacer = SSAReplacer { ssa_ctx: &self.ssa_ctx };
        replacer.visit_stmt_mut(stmt);

        quote::quote! { #stmt }
    }
}

impl<'ast> Visit<'ast> for _Codegen<'ast> {
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        match stmt {
            syn::Stmt::Local(local) => {
                match &local.pat {
                    syn::Pat::Const(_) => todo!("codegen::const"),
                    syn::Pat::Ident(pat_ident) => {
                        let name = pat_ident.ident.to_string();
                        let ssa_name = proc_macro2::Ident::new(
                            &self.ssa_ctx.fresh_name(&name),
                            pat_ident.ident.span()
                        );
                        if !self.to_remove.iter().any(|set| set.contains(&ssa_name)) {
                            if self.fused_codes.contains_key(&ssa_name) {
                                self.push_tokens(self.fused_codes[&ssa_name].clone());
                            } else if let Some(visitor) = self._visitor {
                                if
                                    !visitor.is_tensor_ident(
                                        &proc_macro2::Ident::new(&name, pat_ident.ident.span())
                                    )
                                {
                                    let mut stmt = stmt.clone();
                                    let tokens = self.convert_stmt_to_ssa(&mut stmt);
                                    self.push_tokens(tokens);
                                }
                            }
                        }
                    }
                    syn::Pat::Lit(_) => todo!("codegen::lit"),
                    syn::Pat::Macro(_) => todo!("codegen::macro"),
                    syn::Pat::Or(_) => todo!("codegen::or"),
                    syn::Pat::Paren(_) => todo!("codegen::paren"),
                    syn::Pat::Path(_) => todo!("codegen::path"),
                    syn::Pat::Range(_) => todo!("codegen::range"),
                    syn::Pat::Reference(_) => todo!("codegen::reference"),
                    syn::Pat::Rest(_) => todo!("codegen::rest"),
                    syn::Pat::Slice(_) => todo!("codegen::slice"),
                    syn::Pat::Struct(_) => todo!("codegen::struct"),
                    syn::Pat::Tuple(_) => todo!("codegen::tuple"),
                    syn::Pat::TupleStruct(_) => todo!("codegen::tuple_struct"),
                    syn::Pat::Type(pat_type) => {
                        if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                            let ssa_name = proc_macro2::Ident::new(
                                &self.ssa_ctx.fresh_name(&pat_ident.ident.to_string()),
                                pat_ident.ident.span()
                            );
                            if !self.to_remove.iter().any(|set| set.contains(&ssa_name)) {
                                if self.fused_codes.contains_key(&ssa_name) {
                                    self.push_tokens(self.fused_codes[&ssa_name].clone());
                                }
                            }
                        } else {
                            let mut stmt = stmt.clone();
                            let tokens = self.convert_stmt_to_ssa(&mut stmt);
                            self.push_tokens(tokens);
                        }
                    }
                    syn::Pat::Verbatim(_) => todo!("codegen::verbatim"),
                    syn::Pat::Wild(_) => todo!("codegen::wild"),
                    _ => {
                        let mut stmt = stmt.clone();
                        let tokens = self.convert_stmt_to_ssa(&mut stmt);
                        self.push_tokens(tokens);
                    }
                }
            }
            _ => {
                let mut stmt = stmt.clone();
                let tokens = self.convert_stmt_to_ssa(&mut stmt);
                self.push_tokens(tokens);
            }
        }
    }
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let mut new_codegen = _Codegen {
            fused_codes: self.fused_codes,
            to_remove: self.to_remove,
            current_tokens: Vec::new(),
            ssa_ctx: SSAContext::new(),
            _visitor: if let Some(visitor) = self._visitor {
                visitor.next_visitor.as_ref()
            } else {
                None
            },
            next_codegen: None,
        };
        syn::visit::visit_block(&mut new_codegen, block);
        self.next_codegen = Some(Box::new(new_codegen));
    }
}
