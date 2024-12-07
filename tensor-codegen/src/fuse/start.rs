use std::collections::{ HashMap, HashSet };
use crate::fuse::{ errors::Error, ty_infer::TyInfer };
use petgraph::{ algo::dominators::Dominators, graph::NodeIndex };
use syn::{spanned::Spanned, visit::Visit};
use super::cfg::CFG;

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
    let mut builder = crate::fuse::cfg_builder::CFGBuilder::new(&mut cfg);
    builder.visit_item_fn(item_fn);
    cfg.block_id = core::mem::take(&mut builder.block_ids);
    cfg.fill_variables();
    let dominators = petgraph::algo::dominators::simple_fast(&cfg.graph, cfg.entry);
    let dominance_frontiers = compute_dominance_frontiers(&cfg, &dominators);
    let definitions = cfg.get_variable_definitions();
    cfg.insert_phi_functions(&dominance_frontiers, &definitions);
    cfg.rename_variables(&dominators)?;
    Ok(cfg)
}

pub fn fuse_impl(func: syn::ItemFn) -> anyhow::Result<proc_macro2::TokenStream> {
    let mut cfg = build_cfg(&func)?;
    if !cfg.errors.is_empty() {
        let mut errs = cfg.errors[0].to_syn_error();
        for err in cfg.errors.iter().skip(1) {
            errs.combine(err.to_syn_error());
        }
        return Err(errs.into());
    }
    let mut type_table = TyInfer::new();
    type_table.infer(&cfg)?;
    cfg.live_analysis(&type_table.table);
    cfg.inter_live_analysis();
    println!("cfg: {:#?}", cfg.graph);
    let table = core::mem::take(&mut type_table.table);
    let graphs = cfg.build_graphs(table);
    let mut all_errs = Vec::new();
    for err in graphs.node_weights().map(|x| x.errors.iter()) {
        for e in err {
            all_errs.push(e.to_syn_error());
        }
    }
    if !all_errs.is_empty() {
        let mut first = all_errs[0].clone();
        for e in all_errs.iter().skip(1) {
            first.combine(e.clone());
        }
        return Err(first.into());
    }
    cfg.add_extra_temps(&graphs);
    let mut genfuse_map = HashMap::new();
    for idx in graphs.node_indices() {
        let graph = graphs.node_weight(idx).expect("graph weight not found");
        let petgraph = graph.to_petgraph();
        if petgraph.node_count() > 0 && !petgraph::algo::is_cyclic_directed(&petgraph) {
            println!("graph: {:#?}", petgraph);
            let mut fusion_group = crate::fuse::fuse::fuse(&cfg, &petgraph);
            println!("fusion_group: {:#?}", fusion_group);
            fusion_group.vars.retain(|x| x.0.len() > 1);
            let genfuse = crate::fuse::gen_fuse::gen_fuse(&cfg, &petgraph, &fusion_group);
            let mut stmt_to_remove = Vec::new();
            let mut intermediates = Vec::new();
            for group in &fusion_group.vars {
                stmt_to_remove.push(
                    group.0
                        .iter()
                        .map(|idx| petgraph[*idx].1)
                        .collect::<Vec<_>>()
                );
                intermediates.push(
                    group.0
                        .iter()
                        .map(|idx| *idx)
                        .collect::<Vec<_>>()
                );
            }
            for (i, (_, inps, oust)) in fusion_group.vars.iter().enumerate() {
                for inp in inps.iter() {
                    let pos = intermediates[i].iter().position(|x| *x == inp.comp_graph_idx);
                    if let Some(pos) = pos {
                        intermediates[i].remove(pos);
                    }
                    let pos = stmt_to_remove[i].iter().position(|x| *x == inp.stmt_index);
                    if let Some(pos) = pos {
                        stmt_to_remove[i].remove(pos);
                    }
                }
                for out in oust.iter() {
                    let pos = intermediates[i].iter().position(|x| *x == out.comp_graph_idx);
                    if let Some(pos) = pos {
                        intermediates[i].remove(pos);
                    }
                    let pos = stmt_to_remove[i].iter().position(|x| *x == out.stmt_index);
                    if let Some(pos) = pos {
                        stmt_to_remove[i].remove(pos);
                    }
                }
            }
            println!("stmt_to_remove: {:#?}", stmt_to_remove);
            println!("intermediates: {:#?}", intermediates);
            genfuse_map.insert(idx, (genfuse, fusion_group, stmt_to_remove, intermediates));
        }
    }

    for (idx, (codes, fusion_group, stmt_to_remove, intermediates)) in genfuse_map {
        for (((code, (_, _, out)), remove), intermediate) in codes
            .into_iter()
            .zip(fusion_group.vars.into_iter())
            .zip(stmt_to_remove.into_iter())
            .zip(intermediates.into_iter()) {
            if intermediate.is_empty() {
                continue;
            }
            assert_eq!(out.len(), 1);
            let out = out.iter().next().expect("gen_fuse::output");
            assert_ne!(out.stmt_index, -1);
            if
                let syn::Stmt::Local(local) =
                    &mut cfg.graph[idx].statements[out.stmt_index as usize].stmt
            {
                if let syn::Pat::Ident(ident) = &mut local.pat {
                    ident.ident = out.var.clone();
                } else {
                    return Err(
                        Error::ExpectedIdentifier(local.span(), "fuse_impl").to_anyhow_error()
                    );
                }
                local.init.as_mut().map(|x| {
                    x.expr = Box::new(syn::Expr::Verbatim(code));
                });
            } else {
                cfg.graph[idx].statements[out.stmt_index as usize].stmt = syn::Stmt::Expr(
                    syn::Expr::Verbatim(quote::quote!(#code;)),
                    None
                );
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

    let code = cfg.gen_code();
    let ret = quote::quote!(
        #code
    );
    Ok(ret)
}
