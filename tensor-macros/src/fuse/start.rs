use std::collections::{ HashMap, HashSet };

use crate::fuse::{ codegen::{ Codegen, _Codegen }, ssa::SSAContext, visitor::Visitor };
use syn::visit::Visit;

use crate::fuse::{ dag::Graph, fuse::fuse, gen_fuse::gen_fuse };

pub(crate) fn fuse_impl(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    // println!("func: {:#?}", func);
    let mut visitor = Visitor::new();
    visitor.visit_item_fn(&func);
    if !visitor.visitor.errors.is_empty() {
        // 合并所有错误
        let combined_error = visitor.visitor.errors.into_iter()
            .reduce(|mut acc, e| {
                acc.combine(e);
                acc
            })
            .unwrap();
        return combined_error.to_compile_error().into();
    }
    visitor.remove_unused();
    let graph = Graph::from_nodes(&visitor.visitor.nodes);
    println!("{:#?}", graph);
    let fused = fuse(&graph);
    println!("fused: {:#?}", fused);
    let (fused_codes, fused_outs, fused_inputs) = gen_fuse(&graph, &fused);
    let mut to_remove = vec![];
    // for ((code, input), out) in fused_codes.iter().zip(fused_inputs.iter()).zip(fused_outs.iter()) {
    //     println!(
    //         "input: {:#?}",
    //         input
    //             .iter()
    //             .map(|i| i.to_string())
    //             .collect::<Vec<String>>()
    //     );
    //     println!("code: {:#?}", code.to_string());
    //     println!("out: {:#?}", out.to_string());
    // }

    for ((input, total), out) in fused_inputs.iter().zip(fused.iter()).zip(fused_outs.iter()) {
        let mut intermediate = total
            .iter()
            .map(|i| i.ident.clone())
            .collect::<HashSet<_>>();
        for input in input {
            intermediate.remove(input);
        }
        intermediate.remove(out);
        to_remove.push(intermediate);
    }

    let mut codes = HashMap::new();
    for (i, code) in fused_codes.iter().enumerate() {
        let out = &fused_outs[i];
        codes.insert(out.clone(), quote::quote!(
                let #out = #code;
            ));
    }
    // println!(
    //     "intermediates: {:#?}",
    //     to_remove
    //         .iter()
    //         .map(|i|
    //             i
    //                 .iter()
    //                 .map(|i| i.to_string())
    //                 .collect::<Vec<String>>()
    //         )
    //         .collect::<Vec<Vec<String>>>()
    // );
    // println!("fused_outs: {:#?}", fused_outs);
    let box_visitor = Box::new(visitor.visitor);
    let mut codegen = Codegen {
        _codegen: _Codegen {
            fused_codes: &codes,
            to_remove: &to_remove,
            current_tokens: Vec::new(),
            ssa_ctx: SSAContext::new(),
            _visitor: Some(&box_visitor),
            next_codegen: None,
        },
    };
    codegen.visit_item_fn(&func);
    let code = codegen.get_code();
    println!("code: {:#?}", code.to_string());

    let vis = func.vis.clone();
    let mut sig = func.sig.clone();
    codegen._codegen.convert_signature_to_ssa(&mut sig);
    let ret = quote::quote!(
        #vis #sig {
            #code
        }
    );
    ret.into()
}

pub(crate) fn fuse_proc_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fuse_impl(item)
}
