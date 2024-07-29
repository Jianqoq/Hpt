#![allow(unused_imports)]
use tensor_traits::ops::cmp::TensorCmp;
use maplit::hashmap;
use tensor_common::slice;
use tensor_common::slice::Slice;
use tensor_macros::match_selection;
use tensor_traits::random::Random;
use tensor_traits::shape_manipulate::ShapeManipulate;
use tensor_dyn::slice::SliceOps;
use tensor_traits::tensor::NormalReduce;
use tensor_traits::tensor::TensorCreator;
use tensor_types::dtype::Dtype;
use tensor_traits::ops::uary::FloatUaryOps;
use crate::build::build;
use crate::to_prim_expr::ToPrimeExpr;
use crate::{
    halide::{ code_gen::code_gen::CodeGen, exprs::Int, module::Module, prime_expr::PrimeExpr },
    te::context::Context,
};

#[test]
fn test_reshape_schedule() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let o = ctx.var("o");
    let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let b = ctx.reshape(&a, &[&m, &n, &o, &1i64]);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(outputs.contains(&1));
    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };
    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>
        ::zeros(&[2, 5, 3, 1])
        .expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 1usize => b.clone().into() };
    executable.execute(inps_map, outs_map);
    assert!(a.reshape(&[2, 5, 3, 1]).unwrap().allclose(&b));
}

#[test]
fn test_add_schedule() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let o = ctx.var("o");
    let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let b = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let c = ctx.add(&a, &b);
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 2);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(inputs.contains(&1));
    assert!(outputs.contains(&2));
    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };
    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );

    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let inps_map =
        hashmap! {
            0usize => a.clone().into(),
            1 => b.clone().into(),
        };
    let c = tensor_dyn::tensor::Tensor::<f32>::empty(&[2, 5, 3]).expect("Failed to create tensor");
    let outs_map = hashmap! { 2usize => c.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a + b;
    assert!(test.allclose(&c));
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

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 2);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(inputs.contains(&1));
    assert!(outputs.contains(&2));

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 6.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 1, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 10.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 1])
        .expect("Failed to reshape");
    let inps_map =
        hashmap! {
            0usize => a.clone().into(),
            1 => b.clone().into(),
        };
    let c = tensor_dyn::tensor::Tensor::<f32>::empty(&[2, 5, 3]).expect("Failed to create tensor");
    let outs_map = hashmap! { 2usize => c.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a + b;
    assert!(test.allclose(&c));
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

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 2);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(inputs.contains(&1));
    assert!(outputs.contains(&2));

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 6.0)
        .expect("Failed to create tensor")
        .reshape(&[3, 2, 1])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 10.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5])
        .expect("Failed to reshape");
    let inps_map =
        hashmap! {
            0usize => a.clone().into(),
            1 => b.clone().into(),
        };
    let c = tensor_dyn::tensor::Tensor::<f32>::empty(&[3, 2, 5]).expect("Failed to create tensor");
    let outs_map = hashmap! { 2usize => c.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a + b;
    assert!(test.allclose(&c));
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

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 2);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(inputs.contains(&1));
    assert!(outputs.contains(&3));

    let vars_map = hashmap! {
            "m" => 2,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 6.0)
        .expect("Failed to create tensor")
        .reshape(&[3, 2])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 6.0)
        .expect("Failed to create tensor")
        .reshape(&[3, 2])
        .expect("Failed to reshape");
    let inps_map =
        hashmap! {
            0usize => a.clone().into(),
            1 => b.clone().into(),
        };
    let d = tensor_dyn::tensor::Tensor::<f32>::empty(&[3, 2]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            3usize => d.clone().into(),
        };
    executable.execute(inps_map, outs_map);

    let test = (a + b).sin().unwrap();
    assert!(test.allclose(&d));
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

    let schedule = ctx.to_schedule(&order);
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
                let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1] + 1red1 * istrides0[2]];
                %1_val_ptr[0] = %1_val + %0_val;
            }
            let %1_val = %1_val_ptr[0];
            %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %1_val;
        }
    }
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(outputs.contains(&1));

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::zeros(&[2, 3]).expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 1usize => b.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a.sum([1], false).unwrap();
    assert!(test.allclose(&b));
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

    let schedule = ctx.to_schedule(&order);
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
                let %0_val = %0[3red0 * istrides0[0] + 2red0 * istrides0[1] + 1red0 * istrides0[2]];
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(outputs.contains(&3));

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::zeros(&[1]).expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 3usize => b.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a.sum([0, 1, 2], false).unwrap();
    assert!(test.allclose(&b));
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

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));
    assert!(outputs.contains(&1));

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::zeros(&[1]).expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 1usize => b.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a.sum([0, 1, 2], false).unwrap();
    assert!(test.allclose(&b));
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

    let schedule = ctx.to_schedule(&order);
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
    return;
}"
    ); // prettier-ignore

    let vars_map =
        hashmap! {
            "m" => 2,
            "n" => 5,
            "o" => 3,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 30.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f64>
        ::arange(3.0, 4.0)
        .expect("Failed to create tensor")
        .reshape(&[1])
        .expect("Failed to reshape");
    let inps_map =
        hashmap! {
            0usize => a.clone().into(),
            1 => b.clone().into(),
        };
    let c = tensor_dyn::tensor::Tensor::<f64>::zeros(&[2, 5, 3]).expect("Failed to create tensor");
    let outs_map = hashmap! { 9usize => c.clone().into() };
    executable.execute(inps_map, outs_map);
    let test = &a + &b;
    let sum = test.sum([2], true).unwrap();
    let add = test + &sum;
    let sin = add.sin().unwrap();
    let sum2 = sin.sum([2], true).unwrap();
    let add2 = sin + sum2;
    assert!(add2.allclose(&c));
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
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    for ax0 in range(0, ceil((m - 1) / 2)) {
        for ax1 in range(0, ceil((n - 1) / 2)) {
            let %0_val = %0[(ax0 * (2 * 1) + (0 + 0)) * istrides0[0] + (ax1 * (2 * 1) + (0 + 0)) * istrides0[1]];
            %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %0_val;
        }
    }
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));

    let vars_map = hashmap! {
            "m" => 5,
            "n" => 5,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 25.0)
        .expect("Failed to create tensor")
        .reshape(&[5, 5])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[2, 2]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);
    let test = slice!(a[0:4:2, 0:4:2]).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_slice_nested() {
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
        &b,
        &[
            (&0i64, &(&(4i64).into() - &one), &2i64),
            (&0i64, &(&(4i64).into() - &one), &2i64),
        ]
    );
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();

    assert_eq!(
        func.to_string(),
"fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %2 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, 2) {
        for ax1 in range(0, 2) {
            let %0_val = %0[(ax0 * (2 * (2 * 1)) + (0 + (0 + 0))) * istrides0[0] + (ax1 * (2 * (2 * 1)) + (0 + (0 + 0))) * istrides0[1]];
            let %1_val = %0_val;
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %1_val;
        }
    }
    return;
}"
    ); // prettier-ignore
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));

    let vars_map = hashmap! {
            "m" => 10,
            "n" => 10,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 100.0)
        .expect("Failed to create tensor")
        .reshape(&[10, 10])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[2, 2]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);
    let test = slice!(a[0:9:2, 0:9:2]).unwrap();
    let test = slice!(test[0:4:2, 0:4:2]).unwrap();
    assert!(test.allclose(&d));
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

    let schedule = ctx.to_schedule(&order);
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
            let %1_val_ptr = alloca<f32>(1);
            if (((ax0 >= 5) && (ax0 < 10 + m - 5)) && ((ax1 >= 5) && (ax1 < 10 + n - 5))) {
                let %0_val = %0[(ax0 + (0 - 5)) * istrides0[0] + (ax1 + (0 - 5)) * istrides0[1]];
                %1_val_ptr[0] = %0_val;
            } else {
                %1_val_ptr[0] = 1;
            }
            let %1_val = %1_val_ptr[0];
            let %3_val = sin(%1_val);
            let %2_val = %2[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %4[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %3_val + %2_val;
        }
    }
    return;
}"
    ); // prettier-ignore
}

#[test]
fn test_pad_out() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.pad(
        &a,
        &[
            (&2i64, &2i64),
            (&2i64, &2i64),
        ],
        &1f32
    );
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    for ax0 in range(0, 4 + m) {
        for ax1 in range(0, 4 + n) {
            if (((ax0 >= 2) && (ax0 < 4 + m - 2)) && ((ax1 >= 2) && (ax1 < 4 + n - 2))) {
                let %0_val = %0[(ax0 + (0 - 2)) * istrides0[0] + (ax1 + (0 - 2)) * istrides0[1]];
                %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %0_val;
            } else {
                %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = 1;
            }
        }
    }
    return;
}"
    ); // prettier-ignore

    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert!(inputs.len() == 1);
    assert!(outputs.len() == 1);
    assert!(inputs.contains(&0));

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[8, 8]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);
    println!("{:?}", d);
}

// we need to implement pad slice fusion, need to implemenet a visitor that can rewrite the range of axes
#[test]
fn test_pad_slice() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.pad(
        &a,
        &[
            (&2i64, &2i64),
            (&2i64, &2i64),
        ],
        &1f32
    );
    let c = ctx.slice(
        &b,
        &[
            (&2i64, &6i64, &1i64),
            (&2i64, &6i64, &1i64),
        ]
    );
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
"fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %2 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, 6 - 2) {
        for ax1 in range(0, 6 - 2) {
            let %1_val_ptr = alloca<f32>(1);
            if (((ax0 >= 2) && (ax0 < 4 + m - 2)) && ((ax1 >= 2) && (ax1 < 4 + n - 2))) {
                let %0_val = %0[(ax0 * (1 * 1) + (2 + (0 - 2))) * istrides0[0] + (ax1 * (1 * 1) + (2 + (0 - 2))) * istrides0[1]];
                %1_val_ptr[0] = %0_val;
            } else {
                %1_val_ptr[0] = 1;
            }
            let %1_val = %1_val_ptr[0];
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = %1_val;
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);
    println!("{:?}", d);
}

#[test]
fn test_cast() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.cast(&a, Dtype::F64);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
"fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f64);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            %1[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = (%0_val as f64);
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f64>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.astype::<f64>().unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_prod() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.prod(&a, &1f32, &[1]);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    for ax0 in range(0, m) {
        let %1_val_ptr = alloca<f32>(1);
        %1_val_ptr[0] = 1;
        for 1red1 in range(0, n) {
            let %1_val = %1_val_ptr[0];
            let %0_val = %0[ax0 * istrides0[0] + 1red1 * istrides0[1]];
            %1_val_ptr[0] = %1_val * %0_val;
        }
        let %1_val = %1_val_ptr[0];
        %1[ax0 * ostrides0[0]] = %1_val;
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.prod([1], false).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_min() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.min(&a, &f32::INFINITY, &[1]);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    for ax0 in range(0, m) {
        let %1_val_ptr = alloca<f32>(1);
        %1_val_ptr[0] = inf;
        for 1red1 in range(0, n) {
            let %1_val = %1_val_ptr[0];
            let %0_val = %0[ax0 * istrides0[0] + 1red1 * istrides0[1]];
            %1_val_ptr[0] = min(%1_val, %0_val);
        }
        let %1_val = %1_val_ptr[0];
        %1[ax0 * ostrides0[0]] = %1_val;
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");

    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.min([1], false).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_max() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.max(&a, &f32::NEG_INFINITY, &[1]);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
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
    for ax0 in range(0, m) {
        let %1_val_ptr = alloca<f32>(1);
        %1_val_ptr[0] = -inf;
        for 1red1 in range(0, n) {
            let %1_val = %1_val_ptr[0];
            let %0_val = %0[ax0 * istrides0[0] + 1red1 * istrides0[1]];
            %1_val_ptr[0] = max(%1_val, %0_val);
        }
        let %1_val = %1_val_ptr[0];
        %1[ax0 * ostrides0[0]] = %1_val;
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 16.0)
        .expect("Failed to create tensor")
        .reshape(&[4, 4])
        .expect("Failed to reshape");

    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.max([1], false).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_ge() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.placeholder(&[&m, &1i64], Dtype::F32);
    let c = ctx.ge(&a, &b);
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
        "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *bool);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = (%0_val >= %1_val);
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4, 4]).expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::randn([4, 1]).expect("Failed to reshape");

    let inps_map = hashmap! {
        0usize => a.clone().into(),
        1 => b.clone().into(),
    };
    let d = tensor_dyn::tensor::Tensor::<bool>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.ge(&b).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_gt() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.placeholder(&[&m, &1i64], Dtype::F32);
    let c = ctx.gt(&a, &b);
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
        "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *bool);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = (%0_val > %1_val);
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4, 4]).expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::randn([4, 1]).expect("Failed to reshape");

    let inps_map = hashmap! {
        0usize => a.clone().into(),
        1 => b.clone().into(),
    };
    let d = tensor_dyn::tensor::Tensor::<bool>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.gt(&b).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_le() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.placeholder(&[&m, &1i64], Dtype::F32);
    let c = ctx.le(&a, &b);
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
        "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *bool);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = (%0_val <= %1_val);
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4, 4]).expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::randn([4, 1]).expect("Failed to reshape");

    let inps_map = hashmap! {
        0usize => a.clone().into(),
        1 => b.clone().into(),
    };
    let d = tensor_dyn::tensor::Tensor::<bool>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.le(&b).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_lt() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let a = ctx.placeholder(&[&m, &n], Dtype::F32);
    let b = ctx.placeholder(&[&m, &1i64], Dtype::F32);
    let c = ctx.lt(&a, &b);
    let order = [a.id, b.id, c.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
        "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let istrides1 = istrides_vec[1];
    let %0 = (data_vec[0] as *f32);
    let %1 = (data_vec[1] as *f32);
    let %2 = (output_vec[0] as *bool);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    let n = shape_vars[1];
    for ax0 in range(0, m) {
        for ax1 in range(0, n) {
            let %0_val = %0[ax0 * istrides0[0] + ax1 * istrides0[1]];
            let %1_val = %1[ax0 * istrides1[0] + ax1 * istrides1[1]];
            %2[ax0 * ostrides0[0] + ax1 * ostrides0[1]] = (%0_val < %1_val);
        }
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
            "n" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4, 4]).expect("Failed to reshape");
    let b = tensor_dyn::tensor::Tensor::<f32>::randn([4, 1]).expect("Failed to reshape");

    let inps_map = hashmap! {
        0usize => a.clone().into(),
        1 => b.clone().into(),
    };
    let d = tensor_dyn::tensor::Tensor::<bool>::zeros(&[4, 4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            2 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    let test = a.lt(&b).unwrap();
    assert!(test.allclose(&d));
}

#[test]
fn test_relu() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.relu(&a);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
        "fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    for ax0 in range(0, m) {
        let %0_val = %0[ax0 * istrides0[0]];
        %1[ax0 * ostrides0[0]] = max(%0_val, 0);
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );

    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4]).expect("Failed to reshape");

    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    println!("{:?}", d);
}

#[test]
fn test_selu() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.selu(&a, &1f32);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    assert_eq!(
        func.to_string(),
"fn kernel(istrides_vec: **i64, ostrides_vec: **i64, data_vec: **void, output_vec: **void, offset_vec: *i64, shape_vars: *i64, thread_idx: i64) -> void {
    let istrides0 = istrides_vec[0];
    let %0 = (data_vec[0] as *f32);
    let %1 = (output_vec[0] as *f32);
    let ostrides0 = ostrides_vec[0];
    let m = shape_vars[0];
    for ax0 in range(0, m) {
        let %0_val = %0[ax0 * istrides0[0]];
        %1[ax0 * ostrides0[0]] = ((%0_val > 0) ? %0_val : 1 * (exp(%0_val) - 1));
    }
    return;
}"
    ); // prettier-ignore

    let vars_map = hashmap! {
            "m" => 4,
        };

    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );

    let a = tensor_dyn::tensor::Tensor::<f32>::randn([4]).expect("Failed to reshape");

    let inps_map = hashmap! { 0usize => a.clone().into() };
    let d = tensor_dyn::tensor::Tensor::<f32>::zeros(&[4]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            1 => d.clone().into(),
        };

    executable.execute(inps_map, outs_map);

    println!("{:?}", d);
}
