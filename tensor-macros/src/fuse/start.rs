use std::collections::{ HashMap, HashSet };
use crate::fuse::ty_infer::TyInfer;
use petgraph::{ algo::dominators::Dominators, graph::NodeIndex };
use syn::visit::Visit;
use super::cfg::{ CFGBuilder, CFG };

fn compute_dominance_frontiers(
    cfg: &CFG,
    dominators: &Dominators<NodeIndex>
) -> HashMap<NodeIndex, HashSet<NodeIndex>> {
    let mut dominance_frontiers: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();

    for node in cfg.graph.node_indices() {
        // 跳过根节点，因为根节点没有支配前沿
        if node == dominators.root() {
            continue;
        }

        // 获取当前节点的直接支配者
        let idom = match dominators.immediate_dominator(node) {
            Some(idom) => idom,
            None => {
                continue;
            } // 或者根据需要处理错误
        };

        // 获取当前节点的所有前驱节点
        let preds = cfg.graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .collect::<Vec<_>>();
        if preds.len() >= 2 {
            for pred in preds {
                let mut runner = pred.clone();
                let idom_node = idom.clone();

                while runner != idom_node {
                    // 将当前节点添加到 runner 的支配前沿中
                    dominance_frontiers.entry(runner).or_insert_with(HashSet::new).insert(node);

                    // 更新 runner 为其直接支配者
                    runner = match dominators.immediate_dominator(runner) {
                        Some(dom) => dom,
                        None => {
                            break;
                        } // 或者根据需要处理错误
                    };
                }
            }
        }
    }

    dominance_frontiers
}

fn build_cfg(item_fn: &syn::ItemFn) -> anyhow::Result<CFG> {
    let mut cfg = CFG::new();
    let mut builder = CFGBuilder::new(&mut cfg);
    builder.visit_item_fn(item_fn);
    cfg.block_id = core::mem::take(&mut builder.block_ids);
    let dominators = petgraph::algo::dominators::simple_fast(&cfg.graph, cfg.entry);
    let dominance_frontiers = compute_dominance_frontiers(&cfg, &dominators);
    let definitions = cfg.get_variable_definitions();
    cfg.insert_phi_functions(&dominance_frontiers, &definitions);
    cfg.live_analysis();
    cfg.rename_variables(&dominators);
    Ok(cfg)
}

pub(crate) fn fuse_impl(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let mut cfg = build_cfg(&func).expect("build cfg failed");
    println!("graph: {:#?}", cfg.graph);
    let mut type_table = TyInfer::new();
    type_table.infer(&cfg);
    let graphs = cfg.build_graphs(&type_table.table);

    let mut genfuse_map = HashMap::new();
    for idx in graphs.node_indices() {
        let graph = graphs.node_weight(idx).expect("graph weight not found");
        let petgraph = graph.to_petgraph();
        if petgraph.node_count() > 0 && !petgraph::algo::is_cyclic_directed(&petgraph) {
            let mut fusion_group = crate::fuse::fuse::fuse(&cfg, &petgraph);
            fusion_group.vars.retain(|x| x.len() > 1);
            let genfuse = crate::fuse::gen_fuse::gen_fuse(&cfg, &petgraph, &fusion_group);
            let mut to_remove = Vec::new();
            for group in fusion_group.vars {
                to_remove.push(
                    group
                        .iter()
                        .map(|idx| petgraph[*idx].1)
                        .collect::<Vec<_>>()
                );
            }
            for (i, (inp, out)) in genfuse.1.iter().enumerate() {
                to_remove[i].retain(|v| !inp.iter().any(|(_, stmt_idx, _)| *stmt_idx == *v));
                to_remove[i].retain(|v| !out.iter().any(|(_, stmt_idx, _)| *stmt_idx == *v));
            }
            genfuse_map.insert(idx, (genfuse.0, genfuse.1, to_remove));
        }
    }

    for (idx, (codes, inp_outs, to_remove)) in genfuse_map {
        for ((code, (_, out)), remove) in codes
            .into_iter()
            .zip(inp_outs.into_iter())
            .zip(to_remove.into_iter()) {
            if remove.is_empty() {
                continue;
            }
            assert_eq!(out.len(), 1);
            let (out, out_stmt_idx, _) = &out[0];
            assert_ne!(*out_stmt_idx, -1);
            if
                let syn::Stmt::Local(local) =
                    &mut cfg.graph[idx].statements[*out_stmt_idx as usize].stmt
            {
                if let syn::Pat::Ident(ident) = &mut local.pat {
                    ident.ident = syn::Ident::new(&out.to_string(), out.span());
                } else {
                    panic!("fuse_impl::local::not_ident");
                }
                local.init.as_mut().map(|x| {
                    x.expr = Box::new(syn::Expr::Verbatim(code));
                });
            }
            for &stmt_idx in remove.iter() {
                if stmt_idx >= 0 {
                    cfg.graph[idx].statements[stmt_idx as usize].stmt = syn::Stmt::Expr(
                        syn::Expr::Verbatim(quote::quote!()),
                        None
                    );
                }
            }
        }
    }
    cfg.replace_all_var_back();

    // // process function signature
    let visibility = &func.vis;
    let mut token_stream = proc_macro2::TokenStream::new();
    token_stream.extend(quote::quote!(#visibility));
    let mut signature = func.sig;
    let mut arguments = syn::punctuated::Punctuated::<syn::FnArg, syn::Token![,]>::new();
    let new_inputs = cfg.graph.node_weight((0).into()).expect("node weight not found");
    for inp in new_inputs.statements.iter() {
        if let syn::Stmt::Local(local) = &inp.stmt {
            if let syn::Pat::Type(pat_type) = &local.pat {
                arguments.push(syn::FnArg::Typed(pat_type.clone()));
            } else {
                panic!("fuse_impl::process_function_signature::not_pat_type");
            }
        } else {
            panic!("fuse_impl::process_function_signature::not_local");
        }
    }
    signature.inputs = arguments;
    token_stream.extend(quote::quote!(#signature));
    let body = cfg.gen_code();

    let ret = quote::quote!(
        #token_stream {
            #body
        }
    );
    ret.into()
}

pub(crate) fn fuse_proc_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fuse_impl(item)
}
