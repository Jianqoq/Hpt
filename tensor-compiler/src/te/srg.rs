use std::{ collections::{ HashMap, HashSet }, sync::Arc };

use tensor_common::strides_utils::shape_to_strides;

use crate::{
    halide::{
        exprs::Load,
        let_stmt::LetStmt,
        prime_expr::PrimeExpr,
        stmt::Stmt,
        tensor_load::TensorLoad,
        variable::Variable,
    },
    iter_var::IterVar,
    te::{ hstrides::HStrides, idx_evaluator::IdxEvaluator, stages::{ Body, Stage } },
};

use super::{ rc_mut::RcMut, schedule::Schedule, srg_node::SrgNode, tensor::Tensor };

pub struct Srg {
    pub(crate) nodes: HashMap<usize, SrgNode>,
    pub(crate) tensors: RcMut<HashMap<usize, Tensor>>,
}

impl Srg {
    pub fn create_strides_cal(&mut self, sorted: &[usize]) {
        for id in sorted {
            let inputs = self.nodes[&id].inputs.clone();
            if inputs.len() == 0 {
                let node = self.nodes.get_mut(&id).unwrap();
                let node_shape = node.shape.clone();
                let tensor_sc = self.tensors.borrow().get(&id).unwrap().strides_cal.clone();
                let input_func = Arc::new(move |map: &HashMap<Arc<String>, i64>| {
                    let real_shape = node_shape
                        .iter()
                        .map(|x| { IdxEvaluator::new(map).eval(x) })
                        .collect::<Vec<i64>>();
                    let hstrides = HStrides {
                        strides: shape_to_strides(&real_shape).inner().clone(),
                        reduced_dim: 0,
                        offset: 0,
                    };
                    vec![hstrides]
                });
                let func = tensor_sc(vec![input_func]);
                node.strides_cal = func;
            } else {
                let inputs = self.nodes[&id].inputs.clone();
                let mut input_funcs = vec![];
                for i in inputs.iter() {
                    input_funcs.push(self.nodes[&i].strides_cal.clone());
                }
            }
        }
    }

    pub fn create_schedule(&self, sorted: &[usize]) -> Schedule {
        let mut declared_vars = HashSet::new();
        let mut qa = HashMap::new();
        for id in sorted {
            let node = &self.nodes[id];
            if node.inputs.len() == 0 {
                let body = Body::Stmt(
                    Stmt::LetStmt(
                        LetStmt::make(
                            &Variable::make(&format!("%{}_val", node.id)),
                            TensorLoad {
                                var: Variable::make(&format!("%{}", node.id)).into(),
                                begins: (0..node.shape.len())
                                    .map(|_| (0i64).into())
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                axes: (0..node.shape.len())
                                    .map(|x| Variable::make(&format!("ax{}", x)).into())
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                steps: (0..node.shape.len())
                                    .map(|_| (1i64).into())
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                strides: (0..node.shape.len())
                                    .map(|idx|
                                        Load::make(
                                            Variable::make(&format!("%{}.s", id)),
                                            idx
                                        ).into()
                                    )
                                    .collect::<Vec<PrimeExpr>>()
                                    .into(),
                                hints: vec![].into(),
                            },
                            false,
                            Stmt::None
                        )
                    ).into()
                );
                let stage = Stage {
                    dims: (0..node.shape.len())
                        .map(|x|
                            IterVar::new(0i64, node.shape[x].clone(), 1i64, &format!("ax{}", x))
                        )
                        .collect(),
                    bodys: vec![body],
                    id: *id,
                    out_id: *id,
                    dtype: node.dtype.clone(),
                };
                qa.insert(*id, (Body::Stage(stage), false));
                declared_vars.insert(format!("%{}_val", node.id));
            } else {
                let mut inputs = vec![];
                for i in node.inputs.iter() {
                    inputs.push(qa.get(i).unwrap().0.clone());
                }
                let res = (self.tensors.borrow().get(&node.id).unwrap().body_gen)(
                    inputs,
                    node.is_output(),
                    node.id
                );
                qa.insert(*id, (res, node.is_output()));
            }
        }

        let strides_cal = self.nodes
            .values()
            .filter_map(|node| {
                if node.is_output() { Some(node.strides_cal.clone()) } else { None }
            })
            .last()
            .unwrap();
        Schedule { qa, nodes: self.tensors.clone(), strides_cal }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use maplit::hashmap;
    use tensor_traits::shape_manipulate::ShapeManipulate;
    use tensor_traits::tensor::TensorCreator;
    use tensor_types::dtype::Dtype;

    use crate::{
        halide::{ code_gen::code_gen::CodeGen, exprs::Int, module::Module, prime_expr::PrimeExpr },
        te::context::Context,
        to_prim_expr::ToPrimeExpr,
    };

    #[test]
    fn test_srg_create_strides_cal() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.placeholder(&[&1i64], Dtype::F32);
        let c = ctx.add(&a, &b);
        let d = ctx.sum(&c, &0i64, &[2]);
        let e = ctx.reshape(&d, &[&m, &n, &1i64]);
        let f = ctx.add(&c, &e);
        let g = ctx.sin(&f);
        let h = ctx.sum(&g, &0i64, &[2]);
        let i = ctx.reshape(&h, &[&m, &n, &1i64]);
        let j = ctx.add(&g, &i);
        let order = [a.id, b.id, c.id, d.id, e.id, f.id, g.id, h.id, i.id, j.id];

        let mut srg = ctx.to_srg();
        srg.create_strides_cal(&order);

        let mut var_map = HashMap::new();
        var_map.insert("m".to_string().into(), 1);
        var_map.insert("n".to_string().into(), 8);
        var_map.insert("o".to_string().into(), 8);

        let node = &srg.nodes[order.last().unwrap()];
        let strides = (node.strides_cal)(&var_map);
        println!("{:?}", strides);
    }

    #[test]
    fn test_placeholder() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let a = ctx.placeholder(&[&m, &n], Dtype::F32);
        let order = [a.id];
        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 1);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(outputs.contains(&0));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_reshape_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.reshape(&a, &[&m, &n, &o, &1i64]);
        let order = [a.id, b.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            for ax2 in range(0, o) {
                for ax3 in range(0, 1) {
                    let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1] + ax2 * istrides0[2] + ax3 * istrides0[3]];
                    %1[ax0 * ostrides0[0] + ax1 * ostrides0[1] + ax2 * ostrides0[2] + ax3 * ostrides0[3]] = %0_val;
                }
            }
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 1);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(outputs.contains(&1));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_add_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m], Dtype::F32);
        let b = ctx.placeholder(&[&m], Dtype::F32);
        let c = ctx.add(&a, &b);
        let order = [a.id, b.id, c.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
        assert!(outputs.contains(&2));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
        let vars_map =
            hashmap! {
            "m".to_string().into() => 5,
        };
        let executable = codegen.into_executable(vars_map);

        let a = tensor_dyn::tensor::Tensor::<f32>
            ::arange(0.0, 100.0)
            .expect("Failed to create tensor");
        let b = tensor_dyn::tensor::Tensor::<f32>
            ::arange(0.0, 100.0)
            .expect("Failed to create tensor");
        let inps_map = hashmap! {
            0usize => a.into(),
            1 => b.into(),
        };
        let c = tensor_dyn::tensor::Tensor::<f32>
            ::zeros(&[100])
            .expect("Failed to create tensor");
        let outs_map = hashmap! { 2usize => c.into() };
        executable.execute(inps_map, outs_map);
    }

    #[test]
    fn test_add_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &1i64, &o], Dtype::F32);
        let b = ctx.placeholder(&[&m, &n, &1i64], Dtype::F32);
        let c = ctx.add(&a, &b);
        let order = [a.id, b.id, c.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            for ax2 in range(0, o) {
                let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1] + ax2 * istrides0[2]];
                let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1] + ax2 * istrides1[2]];
                %2[ax0 * ostrides0[0] + ax1 * ostrides0[1] + ax2 * ostrides0[2]] = %0_val + %1_val;
            }
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
        assert!(outputs.contains(&2));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_add_broadcast_diff_len_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&o, &m, &1i64], Dtype::F32);
        let b = ctx.placeholder(&[&m, &n], Dtype::F32);
        let c = ctx.add(&a, &b);
        let order = [a.id, b.id, c.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    for ax0 in range(0, o) {
        for ax1 in range(0, m) {
            for ax2 in range(0, n) {
                let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1] + ax2 * istrides0[2]];
                let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1] + ax2 * istrides1[2]];
                %2[ax0 * ostrides0[0] + ax1 * ostrides0[1] + ax2 * ostrides0[2]] = %0_val + %1_val;
            }
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
        assert!(outputs.contains(&2));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_add_broadcast_diff_len_schedule2() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&o, &m], Dtype::F32);
        let b = ctx.placeholder(&[&o, &m], Dtype::F32);
        let c = ctx.add(&a, &b);
        let d = ctx.sin(&c);
        let order = [a.id, b.id, c.id, d.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %3 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let o = shape_vars[1];
    for ax0 in range(0, o) {
        for ax1 in range(0, m) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1]];
            let %2_val = %0_val + %1_val;
            %3[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = sin(%2_val);
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
        assert!(outputs.contains(&3));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 0);
        codegen.compile();
    }

    #[test]
    fn test_sum_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[1]);
        let order = [a.id, b.id];
        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    for ax0 in range(0, m) {
        for ax1 in range(0, o) {
            let %1_val_ptr = alloca<f32>(1);
            %1_val_ptr[0] = 0;
            for 1red1 in range(0, n) {
                let %1_val = %1_val_ptr[0];
                let %0_val = %0[ax0 * istrides0[0] + 1red1 * istrides0[1] + ax1 * istrides0[2]];
                %1_val_ptr[0] = %1_val + %0_val;
            }
            let %1_val = %1_val_ptr[0];
            %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %1_val;
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 1);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(outputs.contains(&1));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_sum_all_broadcast_schedule() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[0]);
        let c = ctx.sum(&b, &0f32, &[0]);
        let e = ctx.sum(&c, &0f32, &[0]);
        let order = [a.id, b.id, c.id, e.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %3 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    let %3_val_ptr = alloca<f32>(1);
    %3_val_ptr[0] = 0;
    for 3red0 in range(0, o) {
        let %3_val = %3_val_ptr[0];
        let %2_val_ptr = alloca<f32>(1);
        %2_val_ptr[0] = 0;
        for 2red0 in range(0, n) {
            let %2_val = %2_val_ptr[0];
            let %1_val_ptr = alloca<f32>(1);
            %1_val_ptr[0] = 0;
            for 1red0 in range(0, m) {
                let %1_val = %1_val_ptr[0];
                let %0_val = %0[1red0 * istrides0[0] + 2red0 * istrides0[1] + 3red0 * istrides0[2]];
                %1_val_ptr[0] = %1_val + %0_val;
            }
            let %1_val = %1_val_ptr[0];
            %2_val_ptr[0] = %2_val + %1_val;
        }
        let %2_val = %2_val_ptr[0];
        %3_val_ptr[0] = %3_val + %2_val;
    }
    let %3_val = %3_val_ptr[0];
    %3[0] = %3_val;
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 1);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(outputs.contains(&3));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_sum_all_broadcast_schedule2() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.sum(&a, &0f32, &[0, 1, 2]);
        let order = [a.id, b.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    let %1_val_ptr = alloca<f32>(1);
    %1_val_ptr[0] = 0;
    for 1red0 in range(0, m) {
        for 1red1 in range(0, n) {
            for 1red2 in range(0, o) {
                let %1_val = %1_val_ptr[0];
                let %0_val = %0[1red0 * istrides0[0] + 1red1 * istrides0[1] + 1red2 * istrides0[2]];
                %1_val_ptr[0] = %1_val + %0_val;
            }
        }
    }
    let %1_val = %1_val_ptr[0];
    %1[0] = %1_val;
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 1);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(outputs.contains(&1));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }

    #[test]
    fn test_schedule3() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let o = ctx.var("o");
        let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
        let b = ctx.placeholder(&[&1i64], Dtype::F64);
        let c = ctx.add(&a, &b);
        let sum = ctx.sum(&c, &0f32, &[2]);
        let reshaped = ctx.reshape(&sum, &[&m, &n, &1i64]);
        let add = ctx.add(&c, &reshaped);
        let sin = ctx.sin(&add);
        let sum2 = ctx.sum(&sin, &0f32, &[2]);
        let reshaped2 = ctx.reshape(&sum2, &[&m, &n, &1i64]);
        let add2 = ctx.add(&sin, &reshaped2);
        let order = [
            a.id,
            b.id,
            c.id,
            sum.id,
            reshaped.id,
            add.id,
            sin.id,
            sum2.id,
            reshaped2.id,
            add2.id,
        ];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let istrides2 = istrides_vec[2];
    let istrides3 = istrides_vec[3];
    let istrides4 = istrides_vec[4];
    let istrides5 = istrides_vec[5];
    let istrides6 = istrides_vec[6];
    let istrides7 = istrides_vec[7];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f64);
    let %9 = (output_vec[0] as *f64);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    let o = shape_vars[2];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            for ax2 in range(0, o) {
                let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1] + ax2 * istrides0[2]];
                let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1] + ax2 * istrides1[2]];
                let %2_val = %0_val + %1_val;
                let %3_val_ptr = alloca<f64>(1);
                %3_val_ptr[0] = 0;
                for 3red2 in range(0, o) {
                    let %3_val = %3_val_ptr[0];
                    let %0_val = %0[ax0 * istrides2[0] + ax1 * istrides2[1] + ax2 * istrides2[2] + 3red2 * istrides2[3]];
                    let %1_val = %1[ax0 * istrides3[0] + ax1 * istrides3[1] + ax2 * istrides3[2] + 3red2 * istrides3[3]];
                    let %2_val = %0_val + %1_val;
                    %3_val_ptr[0] = %3_val + %2_val;
                }
                let %3_val = %3_val_ptr[0];
                let %4_val = %3_val;
                let %5_val = %2_val + %4_val;
                let %6_val = sin(%5_val);
                let %7_val_ptr = alloca<f64>(1);
                %7_val_ptr[0] = 0;
                for 7red2 in range(0, o) {
                    let %7_val = %7_val_ptr[0];
                    let %0_val = %0[ax0 * istrides4[0] + ax1 * istrides4[1] + ax2 * istrides4[2] + 7red2 * istrides4[3]];
                    let %1_val = %1[ax0 * istrides5[0] + ax1 * istrides5[1] + ax2 * istrides5[2] + 7red2 * istrides5[3]];
                    let %2_val = %0_val + %1_val;
                    let %3_val_ptr = alloca<f64>(1);
                    %3_val_ptr[0] = 0;
                    for 3red2 in range(0, o) {
                        let %3_val = %3_val_ptr[0];
                        let %0_val = %0[ax0 * istrides6[0] + ax1 * istrides6[1] + ax2 * istrides6[2] + 7red2 * istrides6[3] + 3red2 * istrides6[4]];
                        let %1_val = %1[ax0 * istrides7[0] + ax1 * istrides7[1] + ax2 * istrides7[2] + 7red2 * istrides7[3] + 3red2 * istrides7[4]];
                        let %2_val = %0_val + %1_val;
                        %3_val_ptr[0] = %3_val + %2_val;
                    }
                    let %3_val = %3_val_ptr[0];
                    let %4_val = %3_val;
                    let %5_val = %2_val + %4_val;
                    let %6_val = sin(%5_val);
                    %7_val_ptr[0] = %7_val + %6_val;
                }
                let %7_val = %7_val_ptr[0];
                let %8_val = %7_val;
                %9[ax0 * ostrides0[0] + ax1 * ostrides0[1] + ax2 * ostrides0[2]] = %6_val + %8_val;
            }
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&1));
        assert!(outputs.contains(&9));
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 0);
        codegen.compile();
    }

    #[test]
    fn test_slice() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let a = ctx.placeholder(&[&m, &n], Dtype::F32);
        let one = PrimeExpr::Int(Int::make(Dtype::I64, 1));
        let b = ctx.slice(
            &a,
            &[
                (&0i64, &(&m.clone().into() - &one), &2i64),
                (&0i64, &(&n.clone().into() - &one), &2i64),
            ]
        );
        let c = ctx.slice(
            &a,
            &[
                (&1i64, &m, &2i64),
                (&1i64, &n, &2i64),
            ]
        );
        let add = ctx.add(&b, &c);
        let sum = ctx.sum(&add, &0f32, &[0]);
        let order = [a.id, b.id, c.id, add.id, sum.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();
        println!("{}", func);
    }

    #[test]
    fn test_pad() {
        let mut ctx = Context::new();
        let m = ctx.var("m");
        let n = ctx.var("n");
        let a = ctx.placeholder(&[&m, &n], Dtype::F32);
        let b = ctx.pad(
            &a,
            &[
                (&5i64, &5i64),
                (&5i64, &5i64),
            ],
            &1f32
        );
        let c = ctx.placeholder(
            &[&(&m.into() + &(10i64).to_prime_expr()), &(&n.into() + &(10i64).to_prime_expr())],
            Dtype::F32
        );
        let d = ctx.sin(&b);
        let e = ctx.add(&d, &c);
        let order = [a.id, b.id, c.id, d.id, e.id];

        let srg = ctx.to_srg();
        let schedule = srg.create_schedule(&order);
        let func = schedule.to_function();

        assert_eq!(
            func.to_string(),
            "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %2 = (data_vec[1] as *f32);
    let %4 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, 10 + m) {
        for ax1 in range(0, 10 + n) {
            let %1_val = null;
            if (((ax0 >= 5) && (ax0 < 5)) && ((ax1 >= 5) && (ax1 < 5))) {
                let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
                %1_val = %0_val;
            } else {
                %1_val = 1;
            }
            let %3_val = sin(%1_val);
            let %2_val = %2[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %4[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %3_val + %2_val;
        }
    }
}"
        );
        let mut module = Module::new("main");
        let inputs = schedule.inputs();
        let outputs = schedule.outputs();
        assert!(inputs.len() == 2);
        assert!(outputs.len() == 1);
        assert!(inputs.contains(&0));
        assert!(inputs.contains(&2));
        assert!(outputs.contains(&4));
        module.add_function2(&schedule);
        let context = tensor_llvm::context::context::Context::new();
        let mut codegen = CodeGen::new(context, &module, 3);
        codegen.compile();
    }
}
