use std::collections::{ HashMap, HashSet };
use crate::fuse::{ errors::Error, fuse::{ Input, Output }, node::Operand, ty_infer::TyInfer };
use syn::{ spanned::Spanned, visit::Visit };
use super::cfg::CFG;

macro_rules! check_errors {
    ($cfg:expr) => {
        if !$cfg.errors.is_empty() {
            let mut errs = $cfg.errors[0].to_syn_error();
            for err in $cfg.errors.iter().skip(1) {
                errs.combine(err.to_syn_error());
            }
            return Err(errs.into());
        }
    };
}

fn build_cfg(item_fn: &syn::ItemFn) -> anyhow::Result<CFG> {
    let mut cfg = CFG::new();
    let mut builder = crate::fuse::cfg_builder::CFGBuilder::new(&mut cfg);
    builder.visit_item_fn(item_fn);
    let errors = builder.errors.drain(..).collect::<Vec<_>>();
    cfg.block_id = core::mem::take(&mut builder.block_ids);
    cfg.fill_variables();
    let dominators = petgraph::algo::dominators::simple_fast(&cfg.graph, cfg.entry);
    let dominance_frontiers = cfg.compute_dominance_frontiers(&dominators);
    let definitions = cfg.get_variable_definitions();
    cfg.insert_phi_functions(&dominance_frontiers, &definitions);
    cfg.rename_variables(&dominators)?;
    cfg.var_coalescer();
    cfg.errors.extend(errors);
    Ok(cfg)
}

pub fn fuse_impl(func: syn::ItemFn) -> anyhow::Result<proc_macro2::TokenStream> {
    let mut cfg = build_cfg(&func)?;
    check_errors!(cfg);
    // println!("cfg: {:#?}", cfg.graph);
    let mut type_table = TyInfer::new();
    type_table.infer(&cfg)?;
    cfg.live_analysis(&type_table.table);
    cfg.inter_live_analysis();
    let table = core::mem::take(&mut type_table.table);
    let graphs = cfg.build_graphs(table);
    check_errors!(cfg);
    cfg.add_extra_temps(&graphs);
    let mut genfuse_map = HashMap::new();
    for idx in graphs.node_indices() {
        let graph = graphs.node_weight(idx).expect("graph weight not found");
        if graph.nodes.is_empty() {
            continue;
        }
        let cmp_pet_graph = graph.to_cmp_pet_graph();
        // println!("cmp_pet_graph: {:#?}", cmp_pet_graph);
        if cmp_pet_graph.node_count() > 0 && !petgraph::algo::is_cyclic_directed(&cmp_pet_graph) {
            let mut fusion_group = crate::fuse::fuse::cmp_fuse(&cfg, &cmp_pet_graph);
            let mask = fusion_group.groups
                .iter()
                .map(|x| x.len() > 1)
                .collect::<Vec<_>>();
            let mut mask_iter = mask.iter();
            fusion_group.groups.retain(|_| *mask_iter.next().expect("mask_iter"));
            let mut mask_iter = mask.iter();
            fusion_group.inputs.retain(|_| *mask_iter.next().expect("mask_iter"));
            let mut mask_iter = mask.iter();
            fusion_group.stmt_to_remove.retain(|_| *mask_iter.next().expect("mask_iter"));
            let mut mask_iter = mask.iter();
            fusion_group.outputs.retain(|_| *mask_iter.next().expect("mask_iter"));
            let sorted = petgraph::algo::toposort(&cmp_pet_graph, None).expect("toposort failed");
            for (((group, inputs), outputs), stmt_to_remove) in fusion_group.groups
                .iter_mut()
                .zip(fusion_group.inputs.iter_mut())
                .zip(fusion_group.outputs.iter_mut())
                .zip(fusion_group.stmt_to_remove.iter_mut()) {
                let mut sorted_group = Vec::new();
                for sorted_idx in sorted.iter() {
                    if group.contains(sorted_idx) {
                        sorted_group.push(sorted_idx);
                    }
                }
                let mut inserted = HashSet::new();
                let mut intermediates = HashSet::new();
                for idx in sorted_group {
                    let node = &cmp_pet_graph[*idx];
                    if node.args.is_empty() {
                        inputs.insert(Input {
                            var: node.ident.clone(),
                            stmt_index: node.stmt_idx,
                            block_idx: node.block_idx,
                            comp_graph_idx: *idx,
                        });
                        inserted.insert(idx);
                    }
                    for inp in node.args.iter() {
                        if !inserted.contains(inp) && !group.contains(inp) {
                            let inp_node = &cmp_pet_graph[*inp];
                            inputs.insert(Input {
                                var: inp_node.ident.clone(),
                                stmt_index: inp_node.stmt_idx,
                                block_idx: inp_node.block_idx,
                                comp_graph_idx: *inp,
                            });
                            inserted.insert(inp);
                        } else if
                            inserted.contains(inp) &&
                            !inputs.iter().any(|x| x.comp_graph_idx == *inp)
                        {
                            stmt_to_remove.push(cmp_pet_graph[*inp].stmt_idx);
                            intermediates.insert(*inp);
                        }
                    }
                    if !inserted.contains(&idx) {
                        inserted.insert(&idx);
                    }
                }
                for idx in group.iter() {
                    if
                        !inputs.iter().any(|x| x.comp_graph_idx == *idx) &&
                        !intermediates.contains(idx)
                    {
                        outputs.insert(Output {
                            var: cmp_pet_graph[*idx].ident.clone(),
                            stmt_index: cmp_pet_graph[*idx].stmt_idx,
                            block_idx: cmp_pet_graph[*idx].block_idx,
                            comp_graph_idx: *idx,
                        });
                        inserted.insert(idx);
                    }
                }
            }
            // println!("fusion_group: {:#?}", fusion_group);
            let genfuse = crate::fuse::gen_fuse::cmp_gen_fuse(
                &mut cfg,
                &cmp_pet_graph,
                &fusion_group
            );
            check_errors!(cfg);
            genfuse_map.insert(idx, (genfuse, fusion_group));
        }
    }

    let mut func_codes = Vec::new();
    for (idx, (codes, fusion_group)) in genfuse_map {
        for ((code, func_code), ((inp, out), stmt_to_remove)) in codes
            .into_iter()
            .zip(
                fusion_group.inputs
                    .into_iter()
                    .zip(fusion_group.outputs.into_iter())
                    .zip(fusion_group.stmt_to_remove.into_iter())
            ) {
            if stmt_to_remove.is_empty() || inp.is_empty() || out.is_empty() {
                continue;
            }
            assert_eq!(out.len(), 1);
            let out = out.iter().next().expect("gen_fuse::output");
            assert_ne!(out.stmt_index, -1);
            func_codes.push(func_code);
            if
                let syn::Stmt::Local(local) =
                    &mut cfg.graph[idx].statements[out.stmt_index as usize].stmt
            {
                if let syn::Pat::Ident(ident) = &mut local.pat {
                    if let Operand::Variable(var) = &out.var {
                        ident.ident = var.clone();
                    } else {
                        return Err(
                            Error::ExpectedIdentifier(out.var.span(), "fuse_impl").to_anyhow_error()
                        );
                    }
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
            for &stmt_idx in stmt_to_remove.iter() {
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
        #(#func_codes)*
        #code
    );
    Ok(ret)
}
