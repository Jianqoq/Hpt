#![allow(unused_imports)]

use std::sync::Arc;

use tensor_types::dtype::Dtype;

use super::{
    prime_expr::PrimeExpr,
    exprs::{ Int, Load },
    for_stmt::For,
    let_stmt::LetStmt,
    printer::{ self, IRPrinter },
    r#type::{ HalideirTypeCode, Type },
    seq_stmt::Seq,
    stmt::Stmt,
    store_stmt::StoreStmt,
    substitute::subsititue_var::SubstituteVar,
    traits::IRMutateVisitor,
    variable::Variable,
};

#[test]
fn test_halide_expr() {
    let x = Variable::make("x");
    let start = Int::make(Dtype::I64, 0);
    let end = Int::make(Dtype::I64, 10);
    let y = Variable::make("y");
    let i = Variable::make("i");
    let res = Variable::make("res");
    let let_stmt = LetStmt::make(&y, &x + Int::make(Dtype::I64, 1), Stmt::None);
    let load = Load::make_from_strides(&Variable::make("a"), &[i.clone(), x.clone()], &[1, 3]);
    let seq = Seq::make([
        Stmt::LetStmt(let_stmt.clone().into()),
        StoreStmt::make_from_strides(&res, &[&i, &x], &[1, 1], &load).into(),
    ]);
    let for_loop = For::make(&x, &start, &end, 1, &seq);
    let for_loop2 = For::make(&i, &start, &end, 1, &for_loop);
    IRPrinter.print_stmt(&for_loop2);
}

#[test]
fn test_substitue() {
    let x = Variable::make("x");
    let start = Int::make(Dtype::I64, 0);
    let end = Int::make(Dtype::I64, 10);
    let y = Variable::make("y");
    let i = Variable::make("i");
    let res = Variable::make("res");
    let let_stmt = LetStmt::make(&y, &x + Int::make(Dtype::I64, 1), Stmt::None);
    let load = Load::make_from_strides(&Variable::make("a"), &[i.clone(), x.clone()], &[1, 3]);
    let seq = Seq::make([
        Stmt::LetStmt(let_stmt.clone()),
        Stmt::StoreStmt(StoreStmt::make_from_strides(&res, &[&i, &x], &[1, 1], &load)),
    ]);
    let for_loop = For::make(&x, &start, &end, 1, &seq);
    let for_loop2 = For::make(&i, &start, &end, 1, &for_loop);

    let mut substitute = SubstituteVar::new();
    substitute.add_replacement(x, &Variable::make("g"));
    let new = substitute.mutate_stmt(&for_loop2.clone().into());

    IRPrinter.print_stmt(&for_loop2);
    println!("========= replace x with g ============");
    IRPrinter.print_stmt(&new);
}

#[test]
fn test_fusion() {
    // loop1
    let x = Variable::make("x");
    let start = Int::make(Dtype::I64, 0);
    let end = Int::make(Dtype::I64, 512);
    let y = Variable::make("y");
    let start2 = Int::make(Dtype::I64, 0);
    let end2 = Int::make(Dtype::I64, 1);
    let res1 = Variable::make("res1");
    let load = Load::make_from_strides(&Variable::make("a"), &[x.clone(), y.clone()], &[1, 3]);
    let for_loop = For::make(
        &x,
        &start2,
        &end2,
        1,
        StoreStmt::make_from_strides(&res1, &[&x, &y], &[1, 1], &load)
    );
    let for_loop2 = For::make(&y, &start, &end, 1, &for_loop);
    IRPrinter.print_stmt(&for_loop2);

    // loop2
    let z = Variable::make("z");
    let start = Int::make(Dtype::I64, 0);
    let end = Int::make(Dtype::I64, 512);
    let w = Variable::make("w");
    let start2 = Int::make(Dtype::I64, 0);
    let end2 = Int::make(Dtype::I64, 1);
    let res2 = Variable::make("res2");
    let load = Load::make_from_strides(&Variable::make("a"), &[z.clone(), w.clone()], &[1, 3]);
    let for_loop = For::make(
        &z,
        &start,
        &end,
        1,
        StoreStmt::make_from_strides(&res2, &[&z, &w], &[1, 1], &load)
    );
    let for_loop2 = For::make(&w, &start2, &end2, 1, &for_loop);
    IRPrinter.print_stmt(&for_loop2);
}
