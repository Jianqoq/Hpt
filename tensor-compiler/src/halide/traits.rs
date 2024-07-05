use crate::{ hlir::tensor_slice::TensorSlice, iter_var::{ Fused, IterVar, Splitted, _IterVar } };

use super::{
    assign_stmt::AssignStmt,
    exprs::*,
    for_stmt::For,
    if_stmt::IfThenElse,
    inplace_store_stmt::{ InplaceAdd, InplaceDiv, InplaceMul, InplaceStore, InplaceSub },
    let_stmt::LetStmt,
    prime_expr::PrimeExpr,
    seq_stmt::Seq,
    stmt::Stmt,
    store_stmt::StoreStmt,
    variable::Variable,
};

#[allow(unused_variables)]
pub trait IRVisitor where Self: Sized {
    fn visit_expr(&self, expr: &PrimeExpr) {
        match expr {
            PrimeExpr::Int(int) => self.visit_int(&int),
            PrimeExpr::Float(float) => self.visit_float(&float),
            PrimeExpr::UInt(uint) => self.visit_uint(&uint),
            PrimeExpr::Str(string) => self.visit_str(&string),
            PrimeExpr::Variable(var) => self.visit_variable(&var),
            PrimeExpr::Cast(cast) => self.visit_cast(&cast),
            PrimeExpr::Add(add) => self.visit_add(&add),
            PrimeExpr::Sub(sub) => self.visit_sub(&sub),
            PrimeExpr::Mul(mul) => self.visit_mul(&mul),
            PrimeExpr::Div(div) => self.visit_div(&div),
            PrimeExpr::FloorDiv(floor_div) => self.visit_floor_div(&floor_div),
            PrimeExpr::Mod(r#mod) => self.visit_mod(&r#mod),
            PrimeExpr::Min(min) => self.visit_min(&min),
            PrimeExpr::Max(max) => self.visit_max(&max),
            PrimeExpr::Eq(eq) => self.visit_eq(&eq),
            PrimeExpr::Ne(ne) => self.visit_ne(&ne),
            PrimeExpr::Lt(lt) => self.visit_lt(&lt),
            PrimeExpr::Le(le) => self.visit_le(&le),
            PrimeExpr::Gt(gt) => self.visit_gt(&gt),
            PrimeExpr::Ge(ge) => self.visit_ge(&ge),
            PrimeExpr::And(and) => self.visit_and(&and),
            PrimeExpr::Xor(or) => self.visit_xor(&or),
            PrimeExpr::Or(or) => self.visit_or(&or),
            PrimeExpr::Not(not) => self.visit_not(&not),
            PrimeExpr::Call(call) => self.visit_call(&call),
            PrimeExpr::Select(select) => self.visit_select(&select),
            PrimeExpr::Load(load) => self.visit_load(&load),
            PrimeExpr::Let(let_) => self.visit_let(&let_),
            PrimeExpr::Reduce(reduce) => self.visit_reduce(&reduce),
            PrimeExpr::TensorSlice(slice) => self.visit_tensor_slice(&slice),
            PrimeExpr::None => {}
        }
    }
    fn visit_stmt(&self, stmt: &Stmt) {
        match stmt {
            Stmt::LetStmt(let_stmt) => self.visit_let_stmt(&let_stmt),
            Stmt::For(for_stmt) => self.visit_for(&for_stmt),
            Stmt::StoreStmt(store) => self.visit_store(&store),
            Stmt::Seq(stmts) => {
                for stmt in stmts.stmts() {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::IfThenElse(if_then_else) => self.visit_if_then_else(&if_then_else),
            Stmt::InplaceStore(inplace_store) => self.visit_inplace_store(&inplace_store),
            Stmt::InplaceAdd(inplace_add) => self.visit_inplace_add(&inplace_add),
            Stmt::InplaceSub(inplace_sub) => self.visit_inplace_sub(&inplace_sub),
            Stmt::InplaceMul(inplace_mul) => self.visit_inplace_mul(&inplace_mul),
            Stmt::InplaceDiv(inplace_div) => self.visit_inplace_div(&inplace_div),
            Stmt::AssignStmt(assign) => self.visit_assign(&assign),
            Stmt::None => {}
        }
    }
    fn visit_tensor_slice(&self, slice: &TensorSlice) {}
    fn visit_reduce(&self, reduce: &Reduce) {
        reduce
            .expr()
            .iter()
            .for_each(|x| x.accept(self));
        reduce
            .identity()
            .iter()
            .for_each(|x| x.accept(self));
    }
    fn visit_assign(&self, assign: &AssignStmt) {
        assign.lhs().accept(self);
        assign.rhs().accept(self);
    }
    fn visit_variable(&self, var: &Variable) {}
    fn visit_str(&self, string: &Str) {}
    fn visit_cast(&self, cast: &Cast) {
        cast.expr().accept(self);
    }
    fn visit_add(&self, add: &Add) {
        add.e1().accept(self);
        add.e2().accept(self);
    }
    fn visit_sub(&self, sub: &Sub) {
        sub.e1().accept(self);
        sub.e2().accept(self);
    }
    fn visit_mul(&self, mul: &Mul) {
        mul.e1().accept(self);
        mul.e2().accept(self);
    }
    fn visit_div(&self, div: &Div) {
        div.e1().accept(self);
        div.e2().accept(self);
    }
    fn visit_floor_div(&self, floor_div: &FloorDiv) {
        floor_div.e1().accept(self);
        floor_div.e2().accept(self);
    }
    fn visit_mod(&self, mod_: &Mod) {
        mod_.e1().accept(self);
        mod_.e2().accept(self);
    }
    fn visit_min(&self, min: &Min) {
        min.e1().accept(self);
        min.e2().accept(self);
    }
    fn visit_max(&self, max: &Max) {
        max.e1().accept(self);
        max.e2().accept(self);
    }
    fn visit_ge(&self, ge: &Ge) {
        ge.e1().accept(self);
        ge.e2().accept(self);
    }
    fn visit_xor(&self, xor: &Xor) {
        xor.e1().accept(self);
        xor.e2().accept(self);
    }
    fn visit_gt(&self, gt: &Gt) {
        gt.e1().accept(self);
        gt.e2().accept(self);
    }
    fn visit_le(&self, le: &Le) {
        le.e1().accept(self);
        le.e2().accept(self);
    }
    fn visit_lt(&self, lt: &Lt) {
        lt.e1().accept(self);
        lt.e2().accept(self);
    }
    fn visit_eq(&self, eq: &Eq) {
        eq.e1().accept(self);
        eq.e2().accept(self);
    }
    fn visit_ne(&self, ne: &Ne) {
        ne.e1().accept(self);
        ne.e2().accept(self);
    }
    fn visit_and(&self, and: &And) {
        and.e1().accept(self);
        and.e2().accept(self);
    }
    fn visit_or(&self, or: &Or) {
        or.e1().accept(self);
        or.e2().accept(self);
    }
    fn visit_not(&self, not: &Not) {
        not.e().accept(self);
    }
    fn visit_let_stmt(&self, let_stmt: &LetStmt) {
        let_stmt.var().accept(self);
        let_stmt.body().accept(self);
    }
    fn visit_let(&self, let_stmt: &Let) {
        let_stmt.name().accept(self);
        let_stmt.e1().accept(self);
    }
    fn visit_for(&self, for_stmt: &For) {
        for_stmt.var().accept(self);
        for_stmt.start().accept(self);
        for_stmt.end().accept(self);
        for_stmt.stmt().accept(self);
    }
    fn visit_int(&self, int: &Int) {}
    fn visit_uint(&self, uint: &UInt) {}
    fn visit_float(&self, float: &Float) {}
    fn visit_call(&self, call: &Call) {
        for arg in call.args() {
            arg.accept(self);
        }
    }
    fn visit_select(&self, select: &Select) {
        select.cond().accept(self);
        select.true_expr().accept(self);
        select.false_expr().accept(self);
    }
    fn visit_load(&self, load: &Load) {
        load.name().accept(self);
        load.indices().accept(self);
    }
    fn visit_store(&self, store: &StoreStmt) {
        store.var().accept(self);
        store.indices().accept(self);
        store.val().accept(self);
    }
    fn visit_if_then_else(&self, if_then_else: &IfThenElse) {
        if_then_else.cond().accept(self);
        if_then_else.then_case().accept(self);
        if_then_else.else_case().accept(self);
    }

    fn visit_inplace_store(&self, inplace_store: &InplaceStore) {
        inplace_store.to_store().accept(self);
        inplace_store.val().accept(self);
    }

    fn visit_inplace_add(&self, inplace_add: &InplaceAdd) {
        inplace_add.to_store().accept(self);
        inplace_add.val().accept(self);
    }

    fn visit_inplace_sub(&self, inplace_sub: &InplaceSub) {
        inplace_sub.to_store().accept(self);
        inplace_sub.val().accept(self);
    }

    fn visit_inplace_mul(&self, inplace_mul: &InplaceMul) {
        inplace_mul.to_store().accept(self);
        inplace_mul.val().accept(self);
    }

    fn visit_inplace_div(&self, inplace_div: &InplaceDiv) {
        inplace_div.to_store().accept(self);
        inplace_div.val().accept(self);
    }
}

#[allow(unused_variables)]
pub trait IRMutVisitor where Self: Sized {
    fn visit_expr(&mut self, expr: &PrimeExpr) {
        match expr {
            PrimeExpr::Int(int) => self.visit_int(&int),
            PrimeExpr::Float(float) => self.visit_float(&float),
            PrimeExpr::UInt(uint) => self.visit_uint(&uint),
            PrimeExpr::Str(string) => self.visit_str(&string),
            PrimeExpr::Variable(var) => self.visit_variable(&var),
            PrimeExpr::Cast(cast) => self.visit_cast(&cast),
            PrimeExpr::Add(add) => self.visit_add(&add),
            PrimeExpr::Sub(sub) => self.visit_sub(&sub),
            PrimeExpr::Mul(mul) => self.visit_mul(&mul),
            PrimeExpr::Div(div) => self.visit_div(&div),
            PrimeExpr::FloorDiv(floor_div) => self.visit_floor_div(&floor_div),
            PrimeExpr::Mod(r#mod) => self.visit_mod(&r#mod),
            PrimeExpr::Min(min) => self.visit_min(&min),
            PrimeExpr::Max(max) => self.visit_max(&max),
            PrimeExpr::Eq(eq) => self.visit_eq(&eq),
            PrimeExpr::Ne(ne) => self.visit_ne(&ne),
            PrimeExpr::Lt(lt) => self.visit_lt(&lt),
            PrimeExpr::Le(le) => self.visit_le(&le),
            PrimeExpr::Gt(gt) => self.visit_gt(&gt),
            PrimeExpr::Ge(ge) => self.visit_ge(&ge),
            PrimeExpr::And(and) => self.visit_and(&and),
            PrimeExpr::Xor(or) => self.visit_xor(&or),
            PrimeExpr::Or(or) => self.visit_or(&or),
            PrimeExpr::Not(not) => self.visit_not(&not),
            PrimeExpr::Call(call) => self.visit_call(&call),
            PrimeExpr::Select(select) => self.visit_select(&select),
            PrimeExpr::Load(load) => self.visit_load(&load),
            PrimeExpr::Let(let_) => self.visit_let(&let_),
            PrimeExpr::Reduce(reduce) => self.visit_reduce(&reduce),
            PrimeExpr::TensorSlice(slice) => self.visit_tensor_slice(&slice),
            PrimeExpr::None => {}
        }
    }
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::LetStmt(let_stmt) => self.visit_let_stmt(&let_stmt),
            Stmt::For(for_stmt) => self.visit_for(&for_stmt),
            Stmt::StoreStmt(store) => self.visit_store(&store),
            Stmt::Seq(stmts) => {
                for stmt in stmts.stmts() {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::IfThenElse(if_then_else) => self.visit_if_then_else(&if_then_else),
            Stmt::InplaceStore(inplace_store) => self.visit_inplace_store(&inplace_store),
            Stmt::InplaceAdd(inplace_add) => self.visit_inplace_add(&inplace_add),
            Stmt::InplaceSub(inplace_sub) => self.visit_inplace_sub(&inplace_sub),
            Stmt::InplaceMul(inplace_mul) => self.visit_inplace_mul(&inplace_mul),
            Stmt::InplaceDiv(inplace_div) => self.visit_inplace_div(&inplace_div),
            Stmt::AssignStmt(assign) => self.visit_assign(&assign),
            Stmt::None => {}
        }
    }
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {}
    fn visit_reduce(&mut self, reduce: &Reduce) {
        reduce
            .expr()
            .iter()
            .for_each(|x| x.accept_mut(self));
        reduce
            .identity()
            .iter()
            .for_each(|x| x.accept_mut(self));
    }
    fn visit_variable(&mut self, var: &Variable) {}
    fn visit_str(&mut self, string: &Str) {}
    fn visit_assign(&mut self, assign: &AssignStmt) {
        assign.lhs().accept_mut(self);
        assign.rhs().accept_mut(self);
    }
    fn visit_cast(&mut self, cast: &Cast) {
        cast.expr().accept_mut(self);
    }
    fn visit_add(&mut self, add: &Add) {
        add.e1().accept_mut(self);
        add.e2().accept_mut(self);
    }
    fn visit_sub(&mut self, sub: &Sub) {
        sub.e1().accept_mut(self);
        sub.e2().accept_mut(self);
    }
    fn visit_mul(&mut self, mul: &Mul) {
        mul.e1().accept_mut(self);
        mul.e2().accept_mut(self);
    }
    fn visit_div(&mut self, div: &Div) {
        div.e1().accept_mut(self);
        div.e2().accept_mut(self);
    }
    fn visit_floor_div(&mut self, floor_div: &FloorDiv) {
        floor_div.e1().accept_mut(self);
        floor_div.e2().accept_mut(self);
    }
    fn visit_mod(&mut self, mod_: &Mod) {
        mod_.e1().accept_mut(self);
        mod_.e2().accept_mut(self);
    }
    fn visit_min(&mut self, min: &Min) {
        min.e1().accept_mut(self);
        min.e2().accept_mut(self);
    }
    fn visit_max(&mut self, max: &Max) {
        max.e1().accept_mut(self);
        max.e2().accept_mut(self);
    }
    fn visit_ge(&mut self, ge: &Ge) {
        ge.e1().accept_mut(self);
        ge.e2().accept_mut(self);
    }
    fn visit_gt(&mut self, gt: &Gt) {
        gt.e1().accept_mut(self);
        gt.e2().accept_mut(self);
    }
    fn visit_le(&mut self, le: &Le) {
        le.e1().accept_mut(self);
        le.e2().accept_mut(self);
    }
    fn visit_lt(&mut self, lt: &Lt) {
        lt.e1().accept_mut(self);
        lt.e2().accept_mut(self);
    }
    fn visit_eq(&mut self, eq: &Eq) {
        eq.e1().accept_mut(self);
        eq.e2().accept_mut(self);
    }
    fn visit_ne(&mut self, ne: &Ne) {
        ne.e1().accept_mut(self);
        ne.e2().accept_mut(self);
    }
    fn visit_and(&mut self, and: &And) {
        and.e1().accept_mut(self);
        and.e2().accept_mut(self);
    }
    fn visit_xor(&mut self, xor: &Xor) {
        xor.e1().accept_mut(self);
        xor.e2().accept_mut(self);
    }
    fn visit_or(&mut self, or: &Or) {
        or.e1().accept_mut(self);
        or.e2().accept_mut(self);
    }
    fn visit_not(&mut self, not: &Not) {
        not.e().accept_mut(self);
    }
    fn visit_let_stmt(&mut self, let_stmt: &LetStmt) {
        let_stmt.var().accept_mut(self);
        let_stmt.body().accept_mut(self);
    }
    fn visit_let(&mut self, let_stmt: &Let) {
        let_stmt.name().accept_mut(self);
        let_stmt.e1().accept_mut(self);
    }
    fn visit_for(&mut self, for_stmt: &For) {
        for_stmt.var().accept_mut(self);
        for_stmt.start().accept_mut(self);
        for_stmt.end().accept_mut(self);
        for_stmt.stmt().accept_mut(self);
    }
    fn visit_int(&mut self, int: &Int) {}
    fn visit_uint(&mut self, uint: &UInt) {}
    fn visit_float(&mut self, float: &Float) {}
    fn visit_call(&mut self, call: &Call) {
        for arg in call.args() {
            arg.accept_mut(self);
        }
    }
    fn visit_select(&mut self, select: &Select) {
        select.cond().accept_mut(self);
        select.true_expr().accept_mut(self);
        select.false_expr().accept_mut(self);
    }
    fn visit_load(&mut self, load: &Load) {
        load.name().accept_mut(self);
        load.indices().accept_mut(self);
    }
    fn visit_store(&mut self, store: &StoreStmt) {
        store.var().accept_mut(self);
        store.indices().accept_mut(self);
        store.val().accept_mut(self);
    }

    fn visit_if_then_else(&mut self, if_then_else: &IfThenElse) {
        if_then_else.cond().accept_mut(self);
        if_then_else.then_case().accept_mut(self);
        if_then_else.else_case().accept_mut(self);
    }

    fn visit_inplace_store(&mut self, inplace_store: &InplaceStore) {
        inplace_store.to_store().accept_mut(self);
        inplace_store.val().accept_mut(self);
    }

    fn visit_inplace_add(&mut self, inplace_add: &InplaceAdd) {
        inplace_add.to_store().accept_mut(self);
        inplace_add.val().accept_mut(self);
    }

    fn visit_inplace_sub(&mut self, inplace_sub: &InplaceSub) {
        inplace_sub.to_store().accept_mut(self);
        inplace_sub.val().accept_mut(self);
    }

    fn visit_inplace_mul(&mut self, inplace_mul: &InplaceMul) {
        inplace_mul.to_store().accept_mut(self);
        inplace_mul.val().accept_mut(self);
    }

    fn visit_inplace_div(&mut self, inplace_div: &InplaceDiv) {
        inplace_div.to_store().accept_mut(self);
        inplace_div.val().accept_mut(self);
    }
}

pub trait MutatorGetSet {
    fn set_expr<T: Into<PrimeExpr>>(&mut self, expr: T);
    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T);
    fn expr(&self) -> &PrimeExpr;
    fn stmt(&self) -> &Stmt;
}

macro_rules! mutate_binop {
    ($self:ident, $op:ident, $T:ident) => {
        let a = $self.mutate_expr($op.e1());
        let b = $self.mutate_expr($op.e2());
        if &a==$op.e1() && &b==$op.e2() {
            $self.set_expr($op);
        } else {
            $self.set_expr($T::make(a, b));
        }
    };
}

pub(crate) fn mutate_expr<V>(visitor: &mut V, expr: &PrimeExpr) -> PrimeExpr
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    if expr.is_none() {
        visitor.set_expr(PrimeExpr::None);
    } else {
        expr.accept_mutate(visitor);
    }
    visitor.set_stmt(Stmt::None);
    return visitor.expr().clone();
}

pub(crate) fn mutate_stmt<V>(visitor: &mut V, stmt: &Stmt) -> Stmt
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    if stmt.is_none() {
        visitor.set_stmt(Stmt::None);
    } else {
        stmt.accept_mutate(visitor);
    }
    visitor.set_expr(PrimeExpr::None);
    return visitor.stmt().clone();
}

pub(crate) fn visit_expr<V>(visitor: &mut V, expr: &PrimeExpr)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    match expr {
        PrimeExpr::Int(int) => visitor.visit_int(&int),
        PrimeExpr::Float(float) => visitor.visit_float(&float),
        PrimeExpr::UInt(uint) => visitor.visit_uint(&uint),
        PrimeExpr::Str(string) => visitor.visit_str(&string),
        PrimeExpr::Variable(var) => visitor.visit_variable(&var),
        PrimeExpr::Cast(cast) => visitor.visit_cast(&cast),
        PrimeExpr::Add(add) => visitor.visit_add(&add),
        PrimeExpr::Sub(sub) => visitor.visit_sub(&sub),
        PrimeExpr::Mul(mul) => visitor.visit_mul(&mul),
        PrimeExpr::Div(div) => visitor.visit_div(&div),
        PrimeExpr::FloorDiv(floor_div) => visitor.visit_floor_div(&floor_div),
        PrimeExpr::Mod(r#mod) => visitor.visit_mod(&r#mod),
        PrimeExpr::Min(min) => visitor.visit_min(&min),
        PrimeExpr::Max(max) => visitor.visit_max(&max),
        PrimeExpr::Eq(eq) => visitor.visit_eq(&eq),
        PrimeExpr::Ne(ne) => visitor.visit_ne(&ne),
        PrimeExpr::Lt(lt) => visitor.visit_lt(&lt),
        PrimeExpr::Le(le) => visitor.visit_le(&le),
        PrimeExpr::Gt(gt) => visitor.visit_gt(&gt),
        PrimeExpr::Ge(ge) => visitor.visit_ge(&ge),
        PrimeExpr::And(and) => visitor.visit_and(&and),
        PrimeExpr::Xor(or) => visitor.visit_xor(&or),
        PrimeExpr::Or(or) => visitor.visit_or(&or),
        PrimeExpr::Not(not) => visitor.visit_not(&not),
        PrimeExpr::Call(call) => visitor.visit_call(&call),
        PrimeExpr::Select(select) => visitor.visit_select(&select),
        PrimeExpr::Load(load) => visitor.visit_load(&load),
        PrimeExpr::Let(let_) => visitor.visit_let(&let_),
        PrimeExpr::Reduce(reduce) => visitor.visit_reduce(&reduce),
        PrimeExpr::TensorSlice(slice) => visitor.visit_tensor_slice(&slice),
        PrimeExpr::None => {}
    }
}

pub(crate) fn visit_stmt<V>(visitor: &mut V, stmt: &Stmt)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    match stmt {
        Stmt::LetStmt(let_stmt) => visitor.visit_let_stmt(&let_stmt),
        Stmt::For(for_stmt) => visitor.visit_for(&for_stmt),
        Stmt::StoreStmt(store) => visitor.visit_store(&store),
        Stmt::Seq(stmts) => {
            visitor.visit_seq_stmt(&stmts);
        }
        Stmt::IfThenElse(if_then_else) => visitor.visit_if_then_else(&if_then_else),
        Stmt::InplaceStore(inplace_store) => visitor.visit_inplace_store(&inplace_store),
        Stmt::InplaceAdd(inplace_add) => visitor.visit_inplace_add(&inplace_add),
        Stmt::InplaceSub(inplace_sub) => visitor.visit_inplace_sub(&inplace_sub),
        Stmt::InplaceMul(inplace_mul) => visitor.visit_inplace_mul(&inplace_mul),
        Stmt::InplaceDiv(inplace_div) => visitor.visit_inplace_div(&inplace_div),
        Stmt::AssignStmt(assign) => visitor.visit_assign(&assign),
        Stmt::None => {}
    }
}

pub(crate) fn visit_variable<V>(visitor: &mut V, var: &Variable)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    visitor.set_expr(var);
}

pub(crate) fn visit_str<V>(visitor: &mut V, string: &Str)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    visitor.set_expr(string);
}

pub(crate) fn visit_cast<V>(visitor: &mut V, cast: &Cast)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let a = mutate_expr(visitor, cast.expr());
    if &a == cast.expr() {
        visitor.set_expr(cast);
    } else {
        visitor.set_expr(Cast::make(a, *cast.dtype()));
    }
}

pub(crate) fn visit_add<V>(visitor: &mut V, add: &Add)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, add, Add);
}

pub(crate) fn visit_sub<V>(visitor: &mut V, sub: &Sub)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, sub, Sub);
}

pub(crate) fn visit_mul<V>(visitor: &mut V, mul: &Mul)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, mul, Mul);
}

pub(crate) fn visit_div<V>(visitor: &mut V, div: &Div)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, div, Div);
}

pub(crate) fn visit_floor_div<V>(visitor: &mut V, floor_div: &FloorDiv)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, floor_div, FloorDiv);
}

pub(crate) fn visit_mod<V>(visitor: &mut V, mod_: &Mod)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, mod_, Mod);
}

pub(crate) fn visit_min<V>(visitor: &mut V, min: &Min)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, min, Min);
}

pub(crate) fn visit_max<V>(visitor: &mut V, max: &Max)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, max, Max);
}

pub(crate) fn visit_ge<V>(visitor: &mut V, ge: &Ge) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, ge, Ge);
}

pub(crate) fn visit_gt<V>(visitor: &mut V, gt: &Gt) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, gt, Gt);
}

pub(crate) fn visit_le<V>(visitor: &mut V, le: &Le) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, le, Le);
}

pub(crate) fn visit_lt<V>(visitor: &mut V, lt: &Lt) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, lt, Lt);
}

pub(crate) fn visit_eq<V>(visitor: &mut V, eq: &Eq) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, eq, Eq);
}

pub(crate) fn visit_ne<V>(visitor: &mut V, ne: &Ne) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, ne, Ne);
}

pub(crate) fn visit_and<V>(visitor: &mut V, and: &And)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, and, And);
}

pub(crate) fn visit_xor<V>(visitor: &mut V, xor: &Xor)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    mutate_binop!(visitor, xor, Xor);
}

pub(crate) fn visit_or<V>(visitor: &mut V, or: &Or) where V: MutatorGetSet + Sized + IRMutateVisitor {
    mutate_binop!(visitor, or, Or);
}

pub(crate) fn visit_not<V>(visitor: &mut V, not: &Not)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let a = mutate_expr(visitor, not.e());
    if &a == not.e() {
        visitor.set_expr(not);
    } else {
        visitor.set_expr(Not::make(a));
    }
}

pub(crate) fn visit_let_stmt<V>(visitor: &mut V, let_stmt: &LetStmt)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let name = let_stmt.var().into();
    let body = let_stmt.body();
    let var = mutate_expr(visitor, &name);
    let val = mutate_expr(visitor, body);
    if &var == &name && &val == body {
        visitor.set_stmt(let_stmt);
    } else {
        if let Some(var) = var.to_variable() {
            visitor.set_stmt(LetStmt::make(&var, val));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", name, var);
            visitor.set_stmt(Stmt::None);
        }
    }
}

pub(crate) fn visit_let<V>(visitor: &mut V, let_stmt: &Let)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let name = let_stmt.name().into();
    let e1 = let_stmt.e1();
    let var = mutate_expr(visitor, &name);
    let val = mutate_expr(visitor, e1);
    if &var == &name && &val == e1 {
        visitor.set_expr(let_stmt);
    } else {
        if let Some(var) = var.to_variable() {
            visitor.set_expr(Let::make(var, val));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", name, var);
            visitor.set_expr(PrimeExpr::None);
        }
    }
}

pub(crate) fn visit_for<V>(visitor: &mut V, for_stmt: &For)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let var = for_stmt.var().into();
    let new_var = visitor.mutate_expr(&var);
    let start = visitor.mutate_expr(for_stmt.start());
    let end = visitor.mutate_expr(for_stmt.end());
    let stmt = visitor.mutate_stmt(for_stmt.stmt());
    if
        &new_var == &var &&
        &start == for_stmt.start() &&
        &end == for_stmt.end() &&
        &stmt == for_stmt.stmt()
    {
        visitor.set_stmt(for_stmt);
    } else {
        if let Some(new_var) = new_var.to_variable() {
            visitor.set_stmt(For::make(new_var, start, end, stmt));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", var, new_var);
            visitor.set_stmt(Stmt::None);
        }
    }
}

pub(crate) fn visit_seq_stmt<V>(visitor: &mut V, seq: &Seq)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let stmts = seq.stmts();
    let mut changed = false;
    let mut new_stmts = Vec::with_capacity(stmts.len());
    for stmt in stmts.iter() {
        let new_stmt = visitor.mutate_stmt(stmt);
        if &new_stmt != stmt {
            changed = true;
        }
        new_stmts.push(new_stmt);
    }
    if !changed {
        visitor.set_stmt(seq);
    } else {
        visitor.set_stmt(Seq::make(new_stmts));
    }
}

pub(crate) fn visit_int<V>(visitor: &mut V, int: &Int)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    visitor.set_expr(int);
}

pub(crate) fn visit_uint<V>(visitor: &mut V, uint: &UInt)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    visitor.set_expr(uint);
}

pub(crate) fn visit_float<V>(visitor: &mut V, float: &Float)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    visitor.set_expr(float);
}

pub(crate) fn visit_call<V>(visitor: &mut V, call: &Call)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let args = call.args();
    let mut changed = false;
    let mut new_args = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let new_arg = visitor.mutate_expr(arg);
        if &new_arg != arg.as_ref() {
            changed = true;
        }
        new_args.push(new_arg);
    }
    if !changed {
        visitor.set_expr(call);
    } else {
        visitor.set_expr(Call::make(call.name().name(), &new_args));
    }
}

pub(crate) fn visit_select<V>(visitor: &mut V, select: &Select)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let cond = visitor.mutate_expr(select.cond());
    let true_value = visitor.mutate_expr(select.true_expr());
    let false_value = visitor.mutate_expr(select.false_expr());
    if
        &cond == select.cond() &&
        &true_value == select.true_expr() &&
        &false_value == select.false_expr()
    {
        visitor.set_expr(select);
    } else {
        visitor.set_expr(Select::make(cond, true_value, false_value));
    }
}

pub(crate) fn visit_load<V>(visitor: &mut V, load: &Load)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let var = visitor.mutate_expr(load.name());
    let indices = visitor.mutate_expr(load.indices());
    if &var == load.name() && &indices == load.indices() {
        visitor.set_expr(load);
    } else {
        visitor.set_expr(Load::make(var, indices));
    }
}

pub(crate) fn visit_store<V>(visitor: &mut V, store: &StoreStmt)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let var = store.var().into();
    let new_var = visitor.mutate_expr(&var);
    let indices = visitor.mutate_expr(store.indices());
    let val = visitor.mutate_expr(store.val());
    if &new_var == &var && &indices == store.indices() && &val == store.val() {
        visitor.set_stmt(store);
    } else {
        if let Some(new_var) = new_var.to_variable() {
            visitor.set_stmt(StoreStmt::make(new_var, indices, val));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", var, new_var);
            visitor.set_stmt(Stmt::None);
        }
    }
}

pub(crate) fn visit_if_then_else<V>(visitor: &mut V, if_then_else: &IfThenElse)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let cond = visitor.mutate_expr(if_then_else.cond());
    let then_case = visitor.mutate_stmt(if_then_else.then_case());
    let else_case = visitor.mutate_stmt(if_then_else.else_case());
    if
        &cond == if_then_else.cond() &&
        &then_case == if_then_else.then_case() &&
        &else_case == if_then_else.else_case()
    {
        visitor.set_stmt(if_then_else);
    } else {
        visitor.set_stmt(IfThenElse::make(cond, then_case, else_case));
    }
}

pub(crate) fn visit_inplace_store<V>(visitor: &mut V, inplace_store: &InplaceStore)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let to_store = visitor.mutate_expr(inplace_store.to_store());
    let val = visitor.mutate_expr(inplace_store.val());
    if &to_store == inplace_store.to_store() && &val == inplace_store.val() {
        visitor.set_stmt(inplace_store);
    } else {
        visitor.set_stmt(InplaceStore::make(to_store, val));
    }
}

pub(crate) fn visit_inplace_add<V>(visitor: &mut V, inplace_add: &InplaceAdd)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let to_store = visitor.mutate_expr(inplace_add.to_store());
    let val = visitor.mutate_expr(inplace_add.val());
    if &to_store == inplace_add.to_store() && &val == inplace_add.val() {
        visitor.set_stmt(inplace_add);
    } else {
        visitor.set_stmt(InplaceAdd::make(to_store, val));
    }
}

pub(crate) fn visit_inplace_sub<V>(visitor: &mut V, inplace_sub: &InplaceSub)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let to_store = visitor.mutate_expr(inplace_sub.to_store());
    let val = visitor.mutate_expr(inplace_sub.val());
    if &to_store == inplace_sub.to_store() && &val == inplace_sub.val() {
        visitor.set_stmt(inplace_sub);
    } else {
        visitor.set_stmt(InplaceSub::make(to_store, val));
    }
}

pub(crate) fn visit_inplace_mul<V>(visitor: &mut V, inplace_mul: &InplaceMul)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let to_store = visitor.mutate_expr(inplace_mul.to_store());
    let val = visitor.mutate_expr(inplace_mul.val());
    if &to_store == inplace_mul.to_store() && &val == inplace_mul.val() {
        visitor.set_stmt(inplace_mul);
    } else {
        visitor.set_stmt(InplaceMul::make(to_store, val));
    }
}

pub(crate) fn visit_inplace_div<V>(visitor: &mut V, inplace_div: &InplaceDiv)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let to_store = visitor.mutate_expr(inplace_div.to_store());
    let val = visitor.mutate_expr(inplace_div.val());
    if &to_store == inplace_div.to_store() && &val == inplace_div.val() {
        visitor.set_stmt(inplace_div);
    } else {
        visitor.set_stmt(InplaceDiv::make(to_store, val));
    }
}

pub(crate) fn visit_assign<V>(visitor: &mut V, assign: &AssignStmt)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let lhs: PrimeExpr = assign.lhs().into();
    let new_lhs = visitor.mutate_expr(&lhs);
    let new_rhs = visitor.mutate_expr(assign.rhs());
    if new_lhs == lhs && &new_rhs == assign.rhs() {
        visitor.set_stmt(assign);
    } else {
        if let Some(lhs) = lhs.to_variable() {
            visitor.set_stmt(AssignStmt::make(lhs, new_rhs));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", assign.lhs(), lhs);
            visitor.set_stmt(Stmt::None);
        }
    }
}

fn mutate_iter_var<V>(visitor: &mut V, var: &IterVar) -> IterVar
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    match var {
        IterVar::IterVar(var) => {
            let new_start = visitor.mutate_expr(var.start());
            let new_end = visitor.mutate_expr(var.end());
            let new_step = visitor.mutate_expr(var.step());
            let new_var = visitor.mutate_expr(&var.var().into());
            if &new_start == var.start() && &new_end == var.end() && &new_step == var.step() {
                IterVar::IterVar(var.clone())
            } else {
                IterVar::IterVar(
                    _IterVar::new(
                        new_start,
                        new_end,
                        new_step,
                        new_var.to_variable().unwrap().clone()
                    )
                )
            }
        }
        IterVar::Splitted(var) => {
            let outer = mutate_iter_var(visitor, &var.outer);
            let inner = mutate_iter_var(visitor, &var.inner);
            let correspond = mutate_expr(visitor, &var.correspond);
            if
                &outer == var.outer.as_ref() &&
                &inner == var.inner.as_ref() &&
                &correspond == &var.correspond
            {
                IterVar::Splitted(var.clone())
            } else {
                IterVar::Splitted(Splitted::new(outer, inner, correspond))
            }
        }
        IterVar::Fused(fused) => {
            let new_iter_var = mutate_iter_var(visitor, &fused.iter_var);
            let corresponds = [
                mutate_expr(visitor, &fused.corresponds[0]),
                mutate_expr(visitor, &fused.corresponds[1]),
            ];
            if
                &new_iter_var == fused.iter_var.as_ref() &&
                &corresponds[0] == &fused.corresponds[0] &&
                &corresponds[1] == &fused.corresponds[1]
            {
                IterVar::Fused(fused.clone())
            } else {
                IterVar::Fused(Fused::new(fused, corresponds))
            }
        }
    }
}

pub(crate) fn visit_reduce<V>(visitor: &mut V, reduce: &Reduce)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let expr = reduce
        .expr()
        .iter()
        .map(|expr| visitor.mutate_expr(expr))
        .collect();
    let identity = reduce
        .identity()
        .iter()
        .map(|identity| visitor.mutate_expr(identity))
        .collect();
    let mut is_diff = false;
    let new_iter_vars = reduce
        .iter_vars()
        .iter()
        .map(|vars| {
            let new = mutate_iter_var(visitor, vars);
            if &new != vars {
                is_diff = true;
            }
            new
        })
        .collect();
    if &expr == reduce.expr() && &identity == reduce.identity() && !is_diff {
        visitor.set_expr(reduce);
    } else {
        visitor.set_expr(Reduce::make(expr, identity, new_iter_vars, reduce.op()));
    }
}

pub(crate) fn visist_tensor_slice<V>(visitor: &mut V, slice: &TensorSlice)
    where V: MutatorGetSet + Sized + IRMutateVisitor
{
    let var = visitor.mutate_expr(&slice.name().into());
    let dims = slice
        .dims()
        .iter()
        .map(|dim| visitor.mutate_expr(dim))
        .collect::<Vec<PrimeExpr>>();
    if &var == &slice.name().into() && &dims == slice.dims() {
        visitor.set_expr(slice);
    } else {
        visitor.set_expr(TensorSlice::make(var.to_variable().unwrap(), dims));
    }
}

pub trait IRMutateVisitor where Self: MutatorGetSet + Sized {
    fn mutate_expr(&mut self, expr: &PrimeExpr) -> PrimeExpr {
        mutate_expr(self, expr)
    }

    fn mutate_stmt(&mut self, stmt: &Stmt) -> Stmt {
        mutate_stmt(self, stmt)
    }
    fn visit_tensor_slice(&mut self, slice: &TensorSlice) {
        visist_tensor_slice(self, slice);
    }
    fn visit_assign(&mut self, assign: &AssignStmt) {
        visit_assign(self, assign);
    }
    fn visit_expr(&mut self, expr: &PrimeExpr) {
        visit_expr(self, expr);
    }
    fn visit_stmt(&mut self, stmt: &Stmt) {
        visit_stmt(self, stmt);
    }
    fn visit_variable(&mut self, var: &Variable) {
        visit_variable(self, var);
    }
    fn visit_str(&mut self, string: &Str) {
        visit_str(self, string);
    }
    fn visit_cast(&mut self, cast: &Cast) {
        visit_cast(self, cast);
    }
    fn visit_add(&mut self, add: &Add) {
        visit_add(self, add);
    }
    fn visit_sub(&mut self, sub: &Sub) {
        visit_sub(self, sub);
    }
    fn visit_mul(&mut self, mul: &Mul) {
        visit_mul(self, mul);
    }
    fn visit_div(&mut self, div: &Div) {
        visit_div(self, div);
    }
    fn visit_floor_div(&mut self, floor_div: &FloorDiv) {
        visit_floor_div(self, floor_div);
    }
    fn visit_mod(&mut self, mod_: &Mod) {
        visit_mod(self, mod_);
    }
    fn visit_min(&mut self, min: &Min) {
        visit_min(self, min);
    }
    fn visit_max(&mut self, max: &Max) {
        visit_max(self, max);
    }
    fn visit_ge(&mut self, ge: &Ge) {
        visit_ge(self, ge);
    }
    fn visit_gt(&mut self, gt: &Gt) {
        visit_gt(self, gt);
    }
    fn visit_le(&mut self, le: &Le) {
        visit_le(self, le);
    }
    fn visit_lt(&mut self, lt: &Lt) {
        visit_lt(self, lt);
    }
    fn visit_eq(&mut self, eq: &Eq) {
        visit_eq(self, eq);
    }
    fn visit_ne(&mut self, ne: &Ne) {
        visit_ne(self, ne);
    }
    fn visit_and(&mut self, and: &And) {
        visit_and(self, and);
    }
    fn visit_xor(&mut self, xor: &Xor) {
        visit_xor(self, xor);
    }
    fn visit_or(&mut self, or: &Or) {
        visit_or(self, or);
    }
    fn visit_not(&mut self, not: &Not) {
        visit_not(self, not);
    }
    fn visit_let_stmt(&mut self, let_stmt: &LetStmt) {
        visit_let_stmt(self, let_stmt);
    }
    fn visit_let(&mut self, let_stmt: &Let) {
        visit_let(self, let_stmt);
    }
    fn visit_seq_stmt(&mut self, seq: &Seq) {
        visit_seq_stmt(self, seq);
    }
    fn visit_for(&mut self, for_stmt: &For) {
        visit_for(self, for_stmt);
    }
    fn visit_int(&mut self, int: &Int) {
        visit_int(self, int);
    }
    fn visit_uint(&mut self, uint: &UInt) {
        visit_uint(self, uint);
    }
    fn visit_float(&mut self, float: &Float) {
        visit_float(self, float);
    }
    fn visit_call(&mut self, call: &Call) {
        visit_call(self, call);
    }
    fn visit_select(&mut self, select: &Select) {
        visit_select(self, select);
    }
    fn visit_load(&mut self, load: &Load) {
        visit_load(self, load);
    }
    fn visit_store(&mut self, store: &StoreStmt) {
        visit_store(self, store);
    }
    fn visit_if_then_else(&mut self, if_then_else: &IfThenElse) {
        visit_if_then_else(self, if_then_else);
    }
    fn visit_inplace_store(&mut self, inplace_store: &InplaceStore) {
        visit_inplace_store(self, inplace_store);
    }
    fn visit_inplace_add(&mut self, inplace_add: &InplaceAdd) {
        visit_inplace_add(self, inplace_add);
    }
    fn visit_inplace_sub(&mut self, inplace_sub: &InplaceSub) {
        visit_inplace_sub(self, inplace_sub);
    }
    fn visit_inplace_mul(&mut self, inplace_mul: &InplaceMul) {
        visit_inplace_mul(self, inplace_mul);
    }
    fn visit_inplace_div(&mut self, inplace_div: &InplaceDiv) {
        visit_inplace_div(self, inplace_div);
    }
    fn visit_reduce(&mut self, reduce: &Reduce) {
        visit_reduce(self, reduce);
    }
}

pub trait Accepter {
    fn accept<V: IRVisitor>(&self, visitor: &V);
}

pub trait AccepterMut {
    fn accept_mut<V: IRMutVisitor>(&self, visitor: &mut V);
}

pub trait AccepterMutate {
    fn accept_mutate<V: IRMutateVisitor>(&self, visitor: &mut V);
}
