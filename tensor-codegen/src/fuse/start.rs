use std::collections::HashMap;
use crate::fuse::{ errors::Error, ty_infer::TyInfer };
use syn::{ spanned::Spanned, visit::Visit };
use super::cfg::CFG;

fn build_cfg(item_fn: &syn::ItemFn) -> anyhow::Result<CFG> {
    let mut cfg = CFG::new();
    let mut builder = crate::fuse::cfg_builder::CFGBuilder::new(&mut cfg);
    builder.visit_item_fn(item_fn);
    cfg.block_id = core::mem::take(&mut builder.block_ids);
    cfg.fill_variables();
    let dominators = petgraph::algo::dominators::simple_fast(&cfg.graph, cfg.entry);
    let dominance_frontiers = cfg.compute_dominance_frontiers(&dominators);
    let definitions = cfg.get_variable_definitions();
    cfg.insert_phi_functions(&dominance_frontiers, &definitions);
    // println!("graph: {:#?}", cfg.graph);
    cfg.rename_variables(&dominators)?;
    cfg.var_coalescer();
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
    println!("graph: {:#?}", cfg.graph);
    let mut type_table = TyInfer::new();
    type_table.infer(&cfg)?;
    cfg.live_analysis(&type_table.table);
    cfg.inter_live_analysis();
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
            // println!("petgraph: {:#?}", petgraph);
            let mut fusion_group = crate::fuse::fuse::fuse(&cfg, &petgraph);
            // println!("fusion_group: {:#?}", fusion_group);
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
            genfuse_map.insert(idx, (genfuse, fusion_group, stmt_to_remove, intermediates));
        }
    }

    for (idx, (codes, fusion_group, stmt_to_remove, intermediates)) in genfuse_map {
        for (((code, (_, inp, out)), remove), intermediate) in codes
            .into_iter()
            .zip(fusion_group.vars.into_iter())
            .zip(stmt_to_remove.into_iter())
            .zip(intermediates.into_iter()) {
            if intermediate.is_empty() || inp.is_empty() || out.is_empty() {
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
