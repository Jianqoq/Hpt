use proc_macro::TokenStream;
use proc_macro2::Literal;
use syn::{
    parse::Parse,
    parse2,
    parse_macro_input,
    Block,
    Expr,
    ExprArray,
    Ident,
    ItemFn,
    Macro,
    Stmt,
    Token,
};
use quote::{ quote, ToTokens };

struct MacroInput {
    attr: ExprArray,
    item: ItemFn,
}

impl Parse for MacroInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attr: ExprArray = input.parse()?;
        input.parse::<Token![,]>()?;
        let item: ItemFn = input.parse()?;
        Ok(MacroInput { attr, item })
    }
}

fn process_stmt_macro(
    mac: &Macro,
    inp_repeat_count: usize,
    res_repeat_count: usize
) -> TokenStream {
    let new_tokens = if mac.path.is_ident("repeat_inp") {
        let args = parse2::<RepeatInpArgs>(mac.tokens.clone()).expect(
            "Failed to parse repeat_inp arguments"
        );
        let arr = (0..inp_repeat_count as i64).map(|i| {
            let literal = Literal::i64_unsuffixed(i);
            quote! { #literal }
        });
        let name = &args.name;
        let is3 = &args.is3;
        let step_width_m = &args.step_width_m;
        quote! {
            repeat_inp!(#name, #is3, #step_width_m, [#(#arr),*])
        }
    } else if mac.path.is_ident("repeat_kernel") {
        let args = parse2::<RepeatInpArgs>(mac.tokens.clone()).expect(
            "Failed to parse repeat_kernel arguments"
        );
        let arr = (0..res_repeat_count as i64).map(|i| {
            let literal = Literal::i64_unsuffixed(i);
            quote! { #literal }
        });
        let name = &args.name;
        let is3 = &args.is3;
        let step_width_m = &args.step_width_m;
        quote! {
            repeat_kernel!(#name, #is3, #step_width_m, [#(#arr),*])
        }
    } else if mac.path.is_ident("repeat_results") {
        let args = parse2::<RepeatResArgs>(mac.tokens.clone()).expect(
            "Failed to parse repeat_results arguments"
        );
        let arr = (0..inp_repeat_count as i64).map(|i| {
            let literal = Literal::usize_unsuffixed(i as usize);
            quote! { #literal }
        });
        let name = &args.name;
        let inp_name = &args.inp_name;
        let kernel_name = &args.kernel_name;
        let tmp = (0..res_repeat_count).map(|k| {
            let arr = arr.clone();
            let literal = Literal::i64_unsuffixed(k as i64);
            quote! { repeat_results!(#name, #inp_name, #kernel_name, #literal, [#(#arr),*]); }
        });
        quote! {
                {#(#tmp)*}
            }
    } else {
        return (quote! { #mac }).into();
    };
    new_tokens.into()
}

fn process_block(block: &mut Block, inp_repeat_count: usize, res_repeat_count: usize) {
    // println!("{}", block.to_token_stream().to_string());
    for stmt in &mut block.stmts {
        match stmt {
            Stmt::Expr(expr, _) => {
                process_expr(expr, inp_repeat_count, res_repeat_count);
            }
            Stmt::Macro(macro_stmt) => {
                *stmt = syn
                    ::parse2(
                        process_stmt_macro(
                            &macro_stmt.mac,
                            inp_repeat_count,
                            res_repeat_count
                        ).into()
                    )
                    .expect("Failed to parse stmt macro1");
            }
            Stmt::Local(local_stmt) => {
                if let Some(init) = &mut local_stmt.init {
                    process_expr(&mut init.expr, inp_repeat_count, res_repeat_count);
                }
            }
            _ => {}
        }
    }
}

fn process_expr(expr: &mut Expr, inp_repeat_count: usize, res_repeat_count: usize) {
    match expr {
        Expr::Block(block_expr) => {
            process_block(&mut block_expr.block, inp_repeat_count, res_repeat_count);
        }
        Expr::ForLoop(for_loop) => {
            process_block(&mut for_loop.body, inp_repeat_count, res_repeat_count);
        }
        Expr::Unsafe(unsafe_expr) => {
            process_block(&mut unsafe_expr.block, inp_repeat_count, res_repeat_count);
        }
        Expr::Macro(macro_stmt) => {
            let token = process_stmt_macro(&macro_stmt.mac, inp_repeat_count, res_repeat_count);
            *macro_stmt = syn::parse2(token.into()).expect("Failed to parse stmt macro2");
        }
        _ => {}
    }
}

fn process_stmt(stmt: &mut Stmt, inp_repeat: usize, res_repeat: usize) {
    match stmt {
        Stmt::Expr(expr, _) => {
            process_expr(expr, inp_repeat, res_repeat);
        }
        Stmt::Item(_) => {}
        Stmt::Macro(macro_stmt) => {
            let token = process_stmt_macro(&macro_stmt.mac, inp_repeat, res_repeat);
            *stmt = syn
                ::parse2(token.into())
                .inspect_err(|e| {
                    println!("{}", e);
                })
                .expect("Failed to parse stmt macro3");
        }
        Stmt::Local(local_stmt) => {
            if let Some(init) = &mut local_stmt.init {
                process_expr(&mut init.expr, inp_repeat, res_repeat);
            }
        }
    }
}

pub(crate) fn conv2d_microkernel_template(inputs: TokenStream) -> TokenStream {
    let macro_input = parse_macro_input!(inputs as MacroInput);
    let inputs = macro_input.attr;
    let original_body = macro_input.item;
    let mut generated_funcs = vec![];
    for input in inputs.elems.iter() {
        if let Expr::Array(array_expr) = input {
            let mut body = original_body.clone();
            let mut fn_name: Option<Ident> = None;
            let mut inp_repeat_count: usize = 1;
            let mut res_repeat_count: usize = 1;

            for (i, expr) in array_expr.elems.iter().enumerate() {
                if i == 0 {
                    if let Expr::Path(path_expr) = expr {
                        fn_name = Some(path_expr.path.get_ident().unwrap().clone());
                    }
                } else if let Expr::Call(call_expr) = expr {
                    let func_name = call_expr.func.to_token_stream().to_string();
                    if let Some(lit_int) = call_expr.args.first() {
                        let count = lit_int
                            .to_token_stream()
                            .to_string()
                            .parse::<usize>()
                            .expect("Failed to parse repeat count");
                        match func_name.as_str() {
                            "inp_repeat" => {
                                inp_repeat_count = count;
                            }
                            "res_repeat" => {
                                res_repeat_count = count;
                            }
                            _ => {}
                        }
                    }
                }
            }
            let ow_block_def = quote! {
                const OW_BLOCK: usize = #res_repeat_count;
            };
            body.block.stmts.insert(0, syn::parse2(ow_block_def).unwrap());
            body.sig.ident = fn_name.clone().unwrap();
            for stmt in &mut body.block.stmts {
                process_stmt(stmt, inp_repeat_count, res_repeat_count);
            }
            let new_func = quote! {
                #body
            };
            generated_funcs.push(new_func);
        }
    }
    let combined_funcs = quote! {
        #(#generated_funcs)*
    };
    combined_funcs.into()
}

struct RepeatInpArgs {
    name: Ident,
    is3: Ident,
    step_width_m: Expr,
}

impl Parse for RepeatInpArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let is3: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let step_width_m: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        input.parse::<Ident>()?;
        Ok(RepeatInpArgs {
            name,
            is3,
            step_width_m,
        })
    }
}

struct RepeatResArgs {
    name: Ident,
    inp_name: Ident,
    kernel_name: Ident,
}

impl Parse for RepeatResArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let inp_name = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;
        let kernel_name = input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;
        input.parse::<Ident>()?;
        input.parse::<Token![,]>()?;
        input.parse::<Ident>()?;
        Ok(RepeatResArgs {
            name,
            inp_name,
            kernel_name,
        })
    }
}
