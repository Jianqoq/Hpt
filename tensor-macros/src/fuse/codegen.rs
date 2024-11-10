use std::collections::{ HashMap, HashSet };

use syn::visit::*;
use crate::TokenStream2;

pub(crate) struct Codegen {
    pub(crate) fused_codes: HashMap<syn::Ident, TokenStream2>,
    pub(crate) to_remove: Vec<HashSet<syn::Ident>>,
    pub(crate) current_tokens: Vec<TokenStream2>,
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
                    if !self.to_remove.iter().any(|set| set.contains(&pat_ident.ident)) {
                        if self.fused_codes.contains_key(&pat_ident.ident) {
                            self.push_tokens(self.fused_codes[&pat_ident.ident].clone());
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
