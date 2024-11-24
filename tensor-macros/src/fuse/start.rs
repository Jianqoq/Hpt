use std::collections::{ HashMap, HashSet };
use quote::ToTokens;
use crate::fuse::{ cfg::rename_variables, gen_fuse::GenFuse, ty_infer::TyInfer };
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
    let dominators = petgraph::algo::dominators::simple_fast(&cfg.graph, cfg.entry);
    // println!("dominators: {:#?}", dominators);
    let dominance_frontiers = compute_dominance_frontiers(&cfg, &dominators);
    // println!("dominance_frontiers: {:#?}", dominance_frontiers);
    let definitions = cfg.get_variable_definitions();
    cfg.insert_phi_functions(&dominance_frontiers, &definitions);
    cfg.live_analysis();
    // println!("rename: {:#?}", cfg.graph);
    rename_variables(&mut cfg, &dominators);
    // println!("rename: {:#?}", cfg.graph);
    // println!("type_table: {:#?}", type_table.table);
    Ok(cfg)
}

pub(crate) fn fuse_impl(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let mut cfg = build_cfg(&func).expect("build cfg failed");
    // cfg.live_analysis();
    let mut type_table = TyInfer::new();
    type_table.infer(&cfg);
    // println!("type_table: {:#?}", type_table.table);
    // let graphs = cfg.build_graphs(&type_table.table);
    // println!("{:#?}", graphs);

    // let mut genfuse_map = Vec::new();
    // for idx in graphs.node_indices() {
    //     let graph = graphs.node_weight(idx).expect("graph weight not found");
    //     let petgraph = graph.to_petgraph();
    //     if petgraph.node_count() > 0 && !petgraph::algo::is_cyclic_directed(&petgraph) {
    //         let fusion_group = crate::fuse::fuse::fuse(&petgraph);
    //         let genfuse = crate::fuse::gen_fuse::gen_fuse(&petgraph, &fusion_group);
    //         let to_remove = crate::fuse::to_remove::gen_to_remove(
    //             &genfuse,
    //             &fusion_group,
    //             &petgraph
    //         );
    //         genfuse_map.push((idx, genfuse, to_remove));
    //     }
    // }

    // for (idx, gen_fuse, to_remove) in genfuse_map {
    //     if let Some(block) = cfg.graph.node_weight_mut(idx) {
    //         let GenFuse { fused_outs, codes, .. } = gen_fuse;
    //         for ((_, out_idx), code) in fused_outs.into_iter().zip(codes.into_values()) {
    //             if out_idx != -1 {
    //                 block.statements
    //                     .get_mut(out_idx as usize)
    //                     .expect("fuse_impl::get_mut::out_idx").stmt = match
    //                     &block.statements
    //                         .get(out_idx as usize)
    //                         .expect("fuse_impl::get_mut::out_idx").stmt
    //                 {
    //                     syn::Stmt::Local(_) => {
    //                         syn::Stmt::Expr(syn::Expr::Verbatim(quote::quote!(let #code)), None)
    //                     }
    //                     _ => { syn::Stmt::Expr(syn::Expr::Verbatim(code.clone()), None) }
    //                 };
    //                 for set in to_remove.to_remove.iter() {
    //                     for (_, idx) in set {
    //                         block.statements
    //                             .get_mut(*idx as usize)
    //                             .expect("fuse_impl::get_mut::idx").stmt = syn::Stmt::Expr(
    //                             syn::Expr::Verbatim(quote::quote!()),
    //                             None
    //                         );
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    println!("{:#?}", cfg.graph);
    // let mut nodes = cfg.graph.node_indices().collect::<Vec<_>>();
    // nodes.sort();
    // let mut token_stream = proc_macro2::TokenStream::new();

    // // process function signature
    // let visibility = &func.vis;
    // token_stream.extend(quote::quote!(#visibility));
    // let mut signature = func.sig;
    // let mut arguments = syn::punctuated::Punctuated::<syn::FnArg, syn::Token![,]>::new();
    // let new_inputs = cfg.graph.node_weight((0).into()).expect("node weight not found");
    // for inp in new_inputs.statements.iter() {
    //     if let syn::Stmt::Local(local) = &inp.stmt {
    //         if let syn::Pat::Type(pat_type) = &local.pat {
    //             arguments.push(syn::FnArg::Typed(pat_type.clone()));
    //         } else {
    //             panic!("fuse_impl::process_function_signature::not_pat_type");
    //         }
    //     } else {
    //         panic!("fuse_impl::process_function_signature::not_local");
    //     }
    // }
    // signature.inputs = arguments;
    // token_stream.extend(quote::quote!(#signature));
    // let mut body = quote::quote!();
    // for idx in nodes.into_iter().skip(1) {
    //     let node = cfg.graph.node_weight(idx).expect("node weight not found");
    //     match node.block_type {
    //         crate::fuse::cfg::BlockType::Normal => {
    //             body.extend(node.statements.iter().map(|stmt| { quote::quote!(#stmt) }));
    //         }
    //         crate::fuse::cfg::BlockType::IfCond => {
    //             let stmt = node.statements.get(0).expect("node::if_cond::stmt");
    //             body.extend(quote::quote!(if #stmt));
    //         }
    //         crate::fuse::cfg::BlockType::IfThen => {
    //             body.extend({
    //                 let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
    //                 quote::quote!({ #(#iter)* })
    //             });
    //         }
    //         crate::fuse::cfg::BlockType::IfElse => {
    //             body.extend({
    //                 let iter = node.statements.iter().map(|stmt| { quote::quote!(#stmt) });
    //                 quote::quote!(else { #(#iter)* })
    //             });
    //         }
    //         crate::fuse::cfg::BlockType::ForInit => {
    //             let stmt = node.statements.get(0).expect("node::for_init::stmt");
    //             if let syn::Stmt::Local(local) = &stmt.stmt {
    //                 if let syn::Pat::Ident(pat_ident) = &local.pat {
    //                     let ident = &pat_ident.ident;
    //                     body.extend(quote::quote!(for #ident));
    //                 } else {
    //                     panic!("fuse_impl::process_function_signature::not_pat_type");
    //                 }
    //             } else {
    //                 panic!("fuse_impl::process_function_signature::not_local");
    //             }
    //         }
    //         crate::fuse::cfg::BlockType::ForBody => todo!(),
    //         crate::fuse::cfg::BlockType::ForCond => todo!(),
    //         crate::fuse::cfg::BlockType::WhileCond => todo!(),
    //         crate::fuse::cfg::BlockType::WhileBody => todo!(),
    //         crate::fuse::cfg::BlockType::LoopBody => todo!(),
    //     }
    // }
    // println!("{:#?}", token_stream.to_string());
    // println!("{:#?}", body.to_string());
    // visitor.remove_unused();
    // let graph = Graph::from_visitor(&visitor.visitor);
    // println!("{:#?}", graph);
    // let fused = fuse_graph(&graph);
    // let gen_fuse = gen_fuse(&graph._graph, &fused);
    // let to_remove = gen_to_remove(&gen_fuse, &fused);

    // let mut codegen = Codegen {
    //     _codegen: _Codegen {
    //         fused_codes: &gen_fuse,
    //         to_remove: &to_remove,
    //         current_tokens: Vec::new(),
    //         ssa_ctx: RCMut::new(SSAContext::new()),
    //         _visitor: &visitor.visitor,
    //         next_codegen: None,
    //         pat_ident_need_remove: false,
    //         pat_ident_is_ret: false,
    //     },
    // };
    // codegen.visit_item_fn(&func);
    // let code = codegen.get_code();

    let ret = quote::quote!(
        // #func
    );
    ret.into()
}

pub(crate) fn fuse_proc_macro(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fuse_impl(item)
}
