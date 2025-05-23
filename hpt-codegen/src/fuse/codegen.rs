use proc_macro2::TokenStream as TokenStream2;
use quote::ToTokens;

pub(crate) fn stmt(node: &crate::fuse::cfg::BasicBlock) -> TokenStream2 {
    let mut body = quote::quote!();
    match &node.block_type {
        crate::fuse::cfg::BlockType::Normal => {
            body.extend(node.statements.iter().map(|stmt| quote::quote!(#stmt)));
        }
        crate::fuse::cfg::BlockType::IfThen | crate::fuse::cfg::BlockType::IfThenEnd => {
            body.extend({
                let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
                quote::quote!(#(#iter)*)
            });
        }
        crate::fuse::cfg::BlockType::IfElseEnd => {
            body.extend({
                let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
                quote::quote!(#(#iter)*)
            });
        }
        crate::fuse::cfg::BlockType::ForInit => {
            let others = node.statements.iter().skip(1);
            body.extend(others.map(|stmt| quote::quote!(#stmt)));
            let stmt = node.statements.get(0).expect("node::for_init::stmt");
            if let syn::Stmt::Local(local) = &stmt.stmt {
                if let syn::Pat::Ident(pat_ident) = &local.pat {
                    let ident = &pat_ident.ident;
                    body.extend(quote::quote!(#ident));
                } else if let syn::Pat::Wild(_) = &local.pat {
                    body.extend(quote::quote!(_));
                } else {
                    panic!(
                        "fuse_impl::process_function_signature::not_pat_type::{}",
                        local.to_token_stream().to_string()
                    );
                }
            } else {
                panic!("fuse_impl::process_function_signature::not_local");
            }
        }
        crate::fuse::cfg::BlockType::WhileCond
        | crate::fuse::cfg::BlockType::IfCond
        | crate::fuse::cfg::BlockType::ForCond => {
            let stmt = node.statements.get(0).expect("node::while_cond::stmt");
            if let syn::Stmt::Expr(expr, _) = &stmt.stmt {
                body.extend(quote::quote!(#expr));
            } else {
                panic!("cfg_builder::gen_code::ConditionBlock must be expr");
            }
        }
        crate::fuse::cfg::BlockType::ExprBlock
        | crate::fuse::cfg::BlockType::FnBody
        | crate::fuse::cfg::BlockType::LoopBody
        | crate::fuse::cfg::BlockType::WhileBody
        | crate::fuse::cfg::BlockType::ForBody
        | crate::fuse::cfg::BlockType::MatchBody => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::MatchArm => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::Assign => {
            if let syn::Stmt::Local(local) = &node.statements[0].stmt {
                let pat = &local.pat;
                body.extend(quote::quote!(#pat));
            } else {
                panic!("cfg_builder::gen_code::BlockType::ExprBlockAssign");
            }
        }
        crate::fuse::cfg::BlockType::ClosureArgs => {
            let iter = node.statements.iter().map(|stmt| {
                if let syn::Stmt::Local(local) = &stmt.stmt {
                    &local.pat
                } else {
                    panic!("cfg_builder::gen_code::BlockType::ClosureArgs");
                }
            });
            body.extend(quote::quote!(#(#iter), *));
        }
        crate::fuse::cfg::BlockType::ClosureBody => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        super::cfg::BlockType::FnArgs => {
            let iter = node.statements.iter().map(|stmt| {
                if let syn::Stmt::Local(local) = &stmt.stmt {
                    &local.pat
                } else {
                    panic!("cfg_builder::gen_code::BlockType::FnArgs");
                }
            });
            body.extend(quote::quote!(#(#iter), *));
        }
        super::cfg::BlockType::FnVisibility(_) => {}
        super::cfg::BlockType::FnName => {
            if let syn::Stmt::Local(local) = &node.statements[0].stmt {
                let pat = &local.pat;
                body.extend(quote::quote!(#pat));
            } else {
                panic!("cfg_builder::gen_code::BlockType::FnArgs");
            }
        }
        super::cfg::BlockType::FnRet(_) => {}
        super::cfg::BlockType::Generics(_) => {}
        super::cfg::BlockType::Where(_) => {}
        super::cfg::BlockType::MatchCond(_) => {}
        super::cfg::BlockType::MatchCase => {
            let iter = node.statements.iter().map(|stmt| {
                if let syn::Stmt::Local(local) = &stmt.stmt {
                    &local.pat
                } else {
                    panic!("cfg_builder::gen_code::BlockType::FnArgs");
                }
            });
            body.extend(quote::quote!(#(#iter), *));
        }
        crate::fuse::cfg::BlockType::AsyncBlock => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::ConstBlock => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::UnsafeBlock => {
            let iter = node.statements.iter().map(|stmt| quote::quote!(#stmt));
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::FnType(_) => {}
    }
    body
}
