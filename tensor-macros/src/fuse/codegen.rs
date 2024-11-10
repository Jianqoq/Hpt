use std::collections::HashSet;

use syn::visit::*;
use crate::TokenStream2;

pub(crate) struct Codegen {
    pub(crate) fused_codes: Vec<TokenStream2>,
    pub(crate) to_remove: Vec<HashSet<syn::Ident>>,
    pub(crate) current_tokens: Vec<TokenStream2>,
    pub(crate) current_idx: usize,
}

impl Codegen {
    fn push_tokens(&mut self, tokens: TokenStream2) {
        self.current_tokens.push(tokens);
    }

    pub(crate) fn get_code(&mut self) -> TokenStream2 {
        self.current_tokens.drain(..).collect::<TokenStream2>()
    }
}

impl<'ast> Visit<'ast> for Codegen {
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        match stmt {
            syn::Stmt::Local(local) => {
                if let syn::Pat::Ident(pat_ident) = &local.pat {
                    if !self.to_remove[self.current_idx].contains(&pat_ident.ident) {
                        let tokens = quote::quote! { #stmt };
                        self.push_tokens(tokens);
                    } else {
                        self.to_remove[self.current_idx].remove(&pat_ident.ident);
                        if
                            self.to_remove[self.current_idx]
                                .iter()
                                .all(|ident| ident.to_string().starts_with("__out")) ||
                            self.to_remove[self.current_idx].is_empty()
                        {
                            self.push_tokens(self.fused_codes[self.current_idx].clone());
                            if self.current_idx + 1 < self.to_remove.len() {
                                self.current_idx += 1;
                            }
                        }
                    }
                } else {
                    let tokens = quote::quote! { #stmt };
                    self.push_tokens(tokens);
                }
            }
            _ => {
                let tokens = quote::quote! { #stmt };
                self.push_tokens(tokens);
            }
        }
    }
}
