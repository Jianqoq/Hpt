#![allow(unused_imports)]
use hashbrown::HashMap;
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ Call, Int, Load }, printer::IRPrinter, variable::Variable },
    registry::{ Closures, MANAGER },
};

use super::{ expr::Expr, exprs::*, func_type::Type, printer::HlirPrinter };
use super::exprs::Call as HirCall;

#[test]
fn test_fusion() {
    let a = Tensor::make(Shape::new([1, 8, 8]).into(), Dtype::BF16, 4);
    let b = Tensor::make(Shape::new([1]).into(), Dtype::BF16, 5);
    let div = Tensor::make_binop("div", a, b, 6);
    let mut comp = Tensor::make_reduce(
        "max",
        &div,
        [Int::make(Dtype::I64, 2)],
        Int::make(Dtype::I64, 0).into(),
        7
    );
    comp.reshape(&Shape::new([1, 8, 1]));
    let sub = Tensor::make_binop("sub", div, comp, 8);
    let exp = Tensor::make_unop("exp", sub, 5);
    let mut sum = Tensor::make_reduce(
        "sum",
        &exp,
        [Int::make(Dtype::I64, 2)],
        Int::make(Dtype::I64, 0).into(),
        9
    );
    sum.reshape(&Shape::new([1, 8, 1]));
    let div = Tensor::make_binop("div", exp, sum, 10);
    let mut saved_exprs = Vec::new();
    let exp_id = div.id();
    let expr = div.lower(true, &mut vec![], &mut saved_exprs);
    saved_exprs.push((Variable::new(format!("%{}", exp_id)).into(), expr));
    for (k, v) in saved_exprs.iter() {
        println!("{}: {}", k, v);
    }
}

#[test]
fn test_let_bind_plus_fusion() {
    let args: [Variable; 0] = [];
    let mut main = Function::make("main", &args, &Type::make_none(), Expr::None);

    let a = Tensor::make(Shape::new([1, 8, 8]).into(), Dtype::BF16, 0);
    let b = Tensor::make(Shape::new([1]).into(), Dtype::BF16, 1);

    let a_var = Variable::make("a");
    let b_var = Variable::make("b");
    let c = Variable::make("c");
    let reshape_var = Variable::make("reshape");
    let max_var = Variable::make("max");
    let d_var = Variable::make("max_val");
    let e = Variable::make("e");
    let f = Variable::make("f");
    let g = Variable::make("g");
    let h = Variable::make("h");
    let i = Value::make(Dtype::I64, 1);

    let let_ = Let::make(
        &a_var,
        &a,
        Let::make(
            &b_var,
            &b,
            Let::make(
                &c,
                Div::make(&a_var, &b_var),
                Let::make(
                    &d_var,
                    HirCall::make(&max_var, &[&c]),
                    Let::make(
                        &e,
                        HirCall::make(&reshape_var, &[&d_var]),
                        Let::make(
                            &f,
                            Add::make(&e, &e),
                            Let::make(
                                g.clone(),
                                HirCall::make(Variable::make("exp"), &[&f]),
                                Let::make(
                                    &h,
                                    If::make(
                                        Gt::make(
                                            Expr::Value(i),
                                            Value::make(Dtype::BF16, 0.0)
                                        ),
                                        Add::make(&g, &f),
                                        Mul::make(&g, &f)
                                    ),
                                    Return::make(&[&h, &g])
                                )
                            )
                        )
                    )
                )
            )
        )
    );

    main.set_body(let_);
    HlirPrinter.print(main)
}
