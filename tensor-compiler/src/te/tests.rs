#![allow(unused_imports)]
use regex::Regex;
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
use tensor_traits::tensor::IndexReduce;

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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
fn test_argmin_broadcast_schedule() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let o = ctx.var("o");
    let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let b = ctx.argmin(&a, &0f32, 1);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let b = tensor_dyn::tensor::Tensor::<i64>::zeros(&[2, 3]).expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 1usize => b.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a.argmin(1, false).unwrap();
    assert!(test.allclose(&b));
}

#[test]
fn test_argmax_broadcast_schedule() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let o = ctx.var("o");
    let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let b = ctx.argmax(&a, &0f32, 1);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let b = tensor_dyn::tensor::Tensor::<i64>::zeros(&[2, 3]).expect("Failed to create tensor");
    let inps_map = hashmap! { 0usize => a.clone().into() };
    let outs_map = hashmap! { 1usize => b.clone().into() };
    executable.execute(inps_map, outs_map);

    let test = a.argmax(1, false).unwrap();
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
    println!("{}", func.to_string());
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

    let inputs = schedule.inputs();
    let outputs = schedule.outputs();
    assert_eq!(inputs.len(), 1);
    assert_eq!(outputs.len(), 1);
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
fn test_elu() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.elu(&a, &1f32);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

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
    let b = ctx.selu(&a);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

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
fn test_celu() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.celu(&a, &1f32);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

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
fn test_gelu() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.gelu(&a);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

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
fn test_neg() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let a = ctx.placeholder(&[&m], Dtype::F32);
    let b = ctx.neg(&a);
    let order = [a.id, b.id];

    let schedule = ctx.to_schedule(&order);

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
fn test_conv() {
    let mut ctx = Context::new();
    let batch = ctx.var("batch");
    let in_channels = ctx.var("in_channels");
    let in_channels_per_group = ctx.var("in_channels_per_group");
    let in_height = ctx.var("in_height");
    let in_width = ctx.var("in_width");
    let out_channels = ctx.var("out_channels");
    let kernel_height = ctx.var("kernel_height");
    let kernel_width = ctx.var("kernel_width");
    let image = ctx.placeholder(&[&batch, &in_channels, &in_height, &in_width], Dtype::F32);
    let kernel = ctx.placeholder(
        &[&out_channels, &in_channels_per_group, &kernel_height, &kernel_width],
        Dtype::F32
    );
    let b = ctx.conv(&image, &kernel, None, None, None, None, None, None);
    let c = ctx.sin(&b);
    let order = [image.id, kernel.id, b.id, c.id];
    let schedule = ctx.to_schedule(&order);
    let func = schedule.to_function();
    println!("{}", func);
}

#[test]
fn test_concat() {
    let mut ctx = Context::new();
    let m = ctx.var("m");
    let n = ctx.var("n");
    let o = ctx.var("o");
    let p = ctx.var("p");
    let i = ctx.var("i");
    let a = ctx.placeholder(&[&m, &n, &o], Dtype::F32);
    let b = ctx.placeholder(&[&m, &p, &o], Dtype::F32);
    let c = ctx.placeholder(&[&m, &i, &o], Dtype::F32);
    let concat = ctx.concat(1, &[a.clone(), b.clone(), c.clone()]);
    let order = [a.id, b.id, c.id, concat.id];
    let schedule = ctx.to_schedule(&order);
    let vars_map = hashmap! {
            "m" => 2,
            "n" => 3,
            "o" => 3,
            "p" => 5,
            "i" => 4,
        };
    
    let executable = build("main", &[schedule], crate::opt_lvl::OptLvl::O3).into_executable(
        vars_map
    );
    let a = tensor_dyn::tensor::Tensor::<f32>
        ::arange(0.0, 18.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 3, 3])
        .expect("Failed to reshape");

    let b = tensor_dyn::tensor::Tensor::<f32>
        ::arange(30.0, 60.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 5, 3])
        .expect("Failed to reshape");

    let c = tensor_dyn::tensor::Tensor::<f32>
        ::arange(60.0, 84.0)
        .expect("Failed to create tensor")
        .reshape(&[2, 4, 3])
        .expect("Failed to reshape");

    let inps_map = hashmap! {
        0usize => a.clone().into(),
        1 => b.clone().into(),
        2 => c.clone().into(),
    };

    let c = tensor_dyn::tensor::Tensor::<f32>::zeros(&[2, 12, 3]).expect("Failed to create tensor");
    let outs_map = hashmap! {
            3 => c.clone().into(),
        };

    executable.execute(inps_map, outs_map);
    println!("{:?}", c);
}
