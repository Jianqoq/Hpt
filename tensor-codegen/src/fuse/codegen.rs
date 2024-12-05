use quote::ToTokens;
use proc_macro2::TokenStream as TokenStream2;

pub(crate) fn stmt(node: &crate::fuse::cfg::BasicBlock) -> TokenStream2 {
    let mut body = quote::quote!();
    match node.block_type {
        crate::fuse::cfg::BlockType::Normal => {
            body.extend(node.statements.iter().map(|stmt| { quote::quote!(#stmt) }));
        }
        crate::fuse::cfg::BlockType::IfCond => {
            let stmt = node.statements.get(0).expect("node::if_cond::stmt");
            body.extend(quote::quote!(#stmt));
        }
        crate::fuse::cfg::BlockType::ElseIfCond => {
            let stmt = node.statements.get(0).expect("node::else_if_cond::stmt");
            body.extend(quote::quote!(#stmt));
        }
        crate::fuse::cfg::BlockType::IfThen | crate::fuse::cfg::BlockType::IfThenEnd => {
            body.extend({
                let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
                quote::quote!(#(#iter)*)
            });
        }
        crate::fuse::cfg::BlockType::IfElseEnd => {
            body.extend({
                let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
                quote::quote!(#(#iter)*)
            });
        }
        crate::fuse::cfg::BlockType::ForInit => {
            let others = node.statements.iter().skip(1);
            body.extend(others.map(|stmt| { quote::quote!(#stmt) }));
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
        crate::fuse::cfg::BlockType::ForBody => {
            let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::ForCond => {
            let stmt = node.statements.get(0).expect("node::for_cond::stmt");
            body.extend(quote::quote!(#stmt));
        }
        crate::fuse::cfg::BlockType::WhileCond => {
            let stmt = node.statements.get(0).expect("node::while_cond::stmt");
            body.extend(quote::quote!(#stmt));
        }
        crate::fuse::cfg::BlockType::WhileBody => {
            let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::LoopBody => {
            let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
            body.extend(quote::quote!(#(#iter)*));
        }
        crate::fuse::cfg::BlockType::ExprBlock => {
            let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
            body.extend(quote::quote!(#(#iter)*));
        }
        | crate::fuse::cfg::BlockType::ExprBlockAssign
        | crate::fuse::cfg::BlockType::IfAssign
        | crate::fuse::cfg::BlockType::ForAssign
        | crate::fuse::cfg::BlockType::WhileAssign
        | crate::fuse::cfg::BlockType::LoopAssign
        | crate::fuse::cfg::BlockType::ClosureAssign => {
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
            let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
            body.extend(quote::quote!(#(#iter)*));
        }
    }
    body
}
