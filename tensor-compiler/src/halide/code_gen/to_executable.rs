use std::{ collections::{ HashMap, HashSet, VecDeque }, ffi::c_void, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;
use tensor_types::dtype::Dtype;

use crate::{
    edges::Edges,
    halide::{ executable::executable::Executable, module::Function, prime_expr::PrimeExpr },
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator },
};

use super::code_gen::CodeGen;

impl CodeGen {
    pub fn into_executable(self, vars_map: HashMap<Arc<String>, i64>) -> Executable {
        let mut edges = Edges::new();
        let mut funcs = HashSet::new();
        let fns = self.halide_module.fns
            .values()
            .map(|fn_meta| {
                (
                    &fn_meta.function,
                    HashSet::from_iter(fn_meta.inputs_order.iter()),
                    HashSet::from_iter(fn_meta.outputs_order.iter()),
                    &fn_meta.outs_shape,
                )
            })
            .collect::<
                Vec<(&Function, HashSet<&usize>, HashSet<&usize>, &Vec<Arc<Vec<PrimeExpr>>>)>
            >();
        for (func, inputs, _, _) in fns.iter() {
            let input_funcs = inputs
                .iter()
                .map(|x|
                    fns
                        .iter()
                        .filter(|(_, _, outputs, _)| { outputs.contains(x) })
                        .map(|(inp_fn, _, _, _)| inp_fn)
                        .collect::<Vec<&&Function>>()
                )
                .flatten()
                .collect::<HashSet<&&Function>>();
            edges.insert(func, input_funcs);
            funcs.insert(func);
        }
        let order = topo(&edges, &funcs)
            .expect("cycle detected")
            .iter()
            .map(|&x| x.name.clone())
            .collect::<Vec<Arc<String>>>();
        let strides_cal_order = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().strides_cal.clone())
            .collect::<Vec<Arc<dyn Fn(&HashMap<Arc<String>, i64>) -> Vec<HStrides>>>>();
        let sorted_outs_shape = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().outs_shape.clone())
            .collect::<Vec<Vec<Arc<Vec<PrimeExpr>>>>>();
        let sorted_inps_shape = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().inps_shape.clone())
            .collect::<Vec<Vec<Arc<Vec<PrimeExpr>>>>>();
        let sorted_vars = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().vars_order.clone())
            .collect::<Vec<Vec<Arc<String>>>>();
        let sorted_inputs = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().inputs_order.clone())
            .collect::<Vec<Vec<usize>>>();
        let sorted_outputs = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().outputs_order.clone())
            .collect::<Vec<Vec<usize>>>();
        let inps_dtypes = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().inputs_dtype.clone())
            .collect::<Vec<Vec<Dtype>>>();
        let outs_dtypes = order
            .iter()
            .map(|x| self.halide_module.fns.get(x).unwrap().outputs_dtype.clone())
            .collect::<Vec<Vec<Dtype>>>();

        let mut touse_vars_layout = vec![];
        let mut istrides_vec_layout = vec![];
        let mut ostrides_vec_layout = vec![];
        let mut strides_layout = vec![];
        let mut ostrides_layouts = vec![];
        let mut istrides = vec![];
        let mut ostrides = vec![];
        let mut vars = vec![];
        let mut sorted_inputs_container = vec![];
        let mut sorted_outputs_container = vec![];
        let mut sorted_ic_layouts = vec![];
        let mut sorted_oc_layouts = vec![];
        let mut sorted_ic_layout = vec![];
        let mut sorted_oc_layout = vec![];
        let mut real_out_shapes = vec![];
        let mut real_inp_shapes = vec![];

        for (idx, strides_cal) in strides_cal_order.iter().enumerate() {
            let _istrides = {
                let strides = strides_cal(&vars_map);
                let mut cached_layout = vec![];
                let layout = std::alloc::Layout
                    ::from_size_align(strides.len() * std::mem::size_of::<*mut i64>(), 8)
                    .unwrap();
                let strides_vec = unsafe {
                    let strides_vec = std::alloc::alloc(layout.clone()) as *mut *mut i64;
                    for (idx, s) in strides.iter().enumerate() {
                        let s_layout = std::alloc::Layout
                            ::from_size_align(s.strides.len() * std::mem::size_of::<i64>(), 8)
                            .unwrap();
                        cached_layout.push(s_layout);
                        strides_vec.add(idx).write(s.to_aligned_ptr());
                    }
                    strides_vec
                };
                strides_layout.push(cached_layout);
                istrides_vec_layout.push(layout);
                strides_vec
            };
            let mut _layouts = vec![];
            let real_shapes = sorted_outs_shape[idx]
                .iter()
                .map(|x| {
                    Arc::new(
                        x
                            .iter()
                            .map(|x| IdxEvaluator::new(&vars_map).eval(x))
                            .collect::<Vec<i64>>()
                    )
                })
                .collect::<Vec<_>>();
            let real_inps_shapes = sorted_inps_shape[idx]
                .iter()
                .map(|x| {
                    Arc::new(
                        x
                            .iter()
                            .map(|x| IdxEvaluator::new(&vars_map).eval(x))
                            .collect::<Vec<i64>>()
                    )
                })
                .collect::<Vec<_>>();
            let _ostrides = {
                let ptrs_layout = std::alloc::Layout
                    ::from_size_align(real_shapes.len() * std::mem::size_of::<*mut i64>(), 8)
                    .unwrap();
                let ptrs = unsafe {
                    let ptrs = std::alloc::alloc(ptrs_layout.clone()) as *mut *mut i64;
                    for (idx, s) in real_shapes.iter().enumerate() {
                        let layout = std::alloc::Layout
                            ::from_size_align(s.len() * std::mem::size_of::<i64>(), 8)
                            .unwrap();
                        let ptr = std::alloc::alloc(layout) as *mut i64;
                        let strides = shape_to_strides(s);
                        for (idx, stride) in strides.iter().enumerate() {
                            ptr.add(idx).write(*stride);
                        }
                        ptrs.add(idx).write(ptr);
                        _layouts.push(layout);
                    }
                    ptrs
                };
                ostrides_vec_layout.push(ptrs_layout);
                ptrs
            };
            real_out_shapes.push(real_shapes);
            real_inp_shapes.push(real_inps_shapes);
            ostrides_layouts.push(_layouts);

            let _vars = {
                let vars = &sorted_vars[idx];
                let layout = std::alloc::Layout
                    ::from_size_align(vars.len() * std::mem::size_of::<i64>(), 8)
                    .unwrap();
                let shape_vars = unsafe {
                    let shape_vars = std::alloc::alloc(layout.clone()) as *mut i64;
                    for (idx, var) in vars.iter().enumerate() {
                        shape_vars.add(idx).write(vars_map[var]);
                    }
                    shape_vars
                };
                touse_vars_layout.push(layout);
                shape_vars
            };

            let _inputs = {
                let layout = std::alloc::Layout
                    ::from_size_align(
                        sorted_inputs[idx].len() * std::mem::size_of::<*mut c_void>(),
                        8
                    )
                    .unwrap();
                let input_container = unsafe {
                    std::alloc::alloc(layout.clone()) as *mut *mut c_void
                };
                sorted_ic_layout.push(layout);
                input_container
            };

            let _outputs = {
                let layout = std::alloc::Layout
                    ::from_size_align(
                        sorted_outputs[idx].len() * std::mem::size_of::<*mut c_void>(),
                        8
                    )
                    .unwrap();
                let output_container = unsafe {
                    std::alloc::alloc(layout.clone()) as *mut *mut c_void
                };
                sorted_oc_layout.push(layout);
                output_container
            };
            sorted_inputs_container.push(_inputs);
            sorted_outputs_container.push(_outputs);
            sorted_ic_layouts.push(sorted_ic_layout.clone());
            sorted_oc_layouts.push(sorted_oc_layout.clone());
            istrides.push(_istrides);
            ostrides.push(_ostrides);
            vars.push(_vars);
        }

        Executable {
            ctx: self.ctx,
            module: self.module,
            builder: self.builder,
            ee: self.ee,
            sorted_fns: order,
            sorted_strides_cal: strides_cal_order,
            sorted_touse_vars: sorted_vars,
            sorted_out_shapes: sorted_outs_shape,
            touse_vars_layout,
            istrides_vec_layout,
            ostrides_vec_layout,
            strides_layout,
            ostrides_layouts,
            istrides,
            ostrides,
            vars,
            vars_map,
            sorted_inputs,
            sorted_outputs,
            sorted_ic_layouts,
            sorted_oc_layouts,
            sorted_inputs_container,
            sorted_outputs_container,
            inps_shapes: real_inp_shapes,
            outs_shapes: real_out_shapes,
            inps_dtypes,
            outs_dtypes,
        }
    }
}

fn topo<'a>(
    edges: &'a Edges<&&Function>,
    nodes: &'a HashSet<&&Function>
) -> Option<VecDeque<&'a &'a Function>> {
    let mut in_degree = HashMap::new();
    let mut queue = VecDeque::new();
    let mut order = VecDeque::new();
    let edges = edges.invert();
    // calculate in degree
    for &func in nodes.iter() {
        in_degree.entry(func).or_insert(0);
        let edges = edges.get(func);
        if let Some(edges) = edges {
            for &target in edges {
                *in_degree.entry(target).or_insert(0) += 1;
            }
        }
    }

    // push nodes with in degree 0 to queue
    for (&node_id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node_id);
        }
    }

    // topological sort
    while let Some(node_id) = queue.pop_front() {
        order.push_back(node_id);
        if let Some(edges) = edges.get(node_id) {
            for &target in edges {
                let degree = in_degree.get_mut(&target).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            }
        }
    }

    // check if there is a cycle
    if order.len() == nodes.len() {
        Some(order)
    } else {
        None // cycle detected
    }
}
