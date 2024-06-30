#![allow(unused_imports)]
use hashbrown::HashMap;
use tensor_common::{ layout::Layout, shape::Shape };
use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ Call, Int, Load }, printer::IRPrinter, variable::Variable },
    registry::{ Closures, MANAGER },
};

use super::{
    expr::Expr,
    exprs::*,
    func_type::Type,
    lowering::HlirLower,
    printer::HlirPrinter,
    substitute::fuse_cmp::FuseComputeNode,
    traits::HlirMutateVisitor,
};
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
        Int::make(Dtype::I64, 1).into(),
        9
    );
    sum.reshape(&Shape::new([1, 8, 1]));
    let div = Tensor::make_binop("div", exp, sum, 10);
    let mut lower = HlirLower::new();
    lower.lower(div);
    lower.print_lowered_fors();
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
    let c_expr: Expr = c.clone().into();
    let reshape_var = Variable::make("reshape");
    let max_var = Variable::make("max");
    let d_var = Variable::make("max_val");
    let d_var_expr: Expr = d_var.clone().into();
    let e = Variable::make("e");
    let f = Variable::make("f");
    let g = Variable::make("g");

    let let_ = Let::make(
        &a_var,
        &a,
        Let::make(
            &b_var,
            &b,
            Let::make(
                &c,
                HirCall::make(Variable::make("div"), &[&a_var, &b_var]),
                Let::make(
                    &d_var,
                    HirCall::make(
                        &max_var,
                        &[
                            &c_expr,
                            &Tuple::make([Int::make(Dtype::I64, 2)]).into(),
                            &Int::make(Dtype::I64, 0).into(),
                        ]
                    ),
                    Let::make(
                        &e,
                        HirCall::make(
                            &reshape_var,
                            &[
                                &d_var_expr,
                                &Tuple::make([
                                    Int::make(Dtype::I64, 1),
                                    Int::make(Dtype::I64, 8),
                                    Int::make(Dtype::I64, 1),
                                ]).into(),
                            ]
                        ),
                        Let::make(
                            &f,
                            HirCall::make(Variable::make("add"), &[&e, &e]),
                            Let::make(
                                g.clone(),
                                HirCall::make(Variable::make("exp"), &[&f]),
                                Return::make(&[&g])
                            )
                        )
                    )
                )
            )
        )
    );

    main.set_body(let_);

    let mut visitor = FuseComputeNode::new();
    visitor.visit_function(&main);
    HlirPrinter.print(main);
    let g = visitor.map().get(&g).unwrap();
    let mut lower = HlirLower::new();
    lower.lower(g);
    lower.print_lowered_fors();
}

#[test]
fn test_slice() {
    let args: [Variable; 0] = [];
    let mut main = Function::make("main", &args, &Type::make_none(), Expr::None);

    let a = Tensor::make(Shape::new([4, 5, 6]).into(), Dtype::BF16, 0);
    let sliced = Variable::make("sliced");

    let let_ = Let::make(&sliced, HirCall::make(Variable::make("slice"), &[&a]), Return::make(&[&sliced]));

    main.set_body(let_);

    let mut visitor = FuseComputeNode::new();
    visitor.visit_function(&main);
    HlirPrinter.print(main);
    let g = visitor.map().get(&sliced).unwrap();
    let mut lower = HlirLower::new();
    lower.lower(g);
    lower.print_lowered_fors();
}
