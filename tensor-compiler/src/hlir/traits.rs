use super::{ exprs::*, node::Expr };

pub trait HlirAcceptor {
    fn accept<V: HlirVisitor>(&self, visitor: &V);
}

pub trait HlirAccepterMut {
    fn accept_mut<V: HlirMutVisitor>(&self, visitor: &mut V);
}

pub trait HlirAccepterMutate {
    fn accept_mutate<V: HlirMutateVisitor>(&self, visitor: &mut V);
}

pub trait HlirVisitor where Self: Sized {
    fn visit_expr(&self, expr: &Expr) {
        match expr {
            Expr::Value(value) => self.visit_value(&value),
            Expr::Str(string) => self.visit_str(&string),
            Expr::Variable(var) => self.visit_variable(&var),
            Expr::Cast(cast) => self.visit_cast(&cast),
            Expr::Add(add) => self.visit_add(&add),
            Expr::Sub(sub) => self.visit_sub(&sub),
            Expr::Mul(mul) => self.visit_mul(&mul),
            Expr::Div(div) => self.visit_div(&div),
            Expr::Mod(r#mod) => self.visit_mod(&r#mod),
            Expr::Min(min) => self.visit_min(&min),
            Expr::Max(max) => self.visit_max(&max),
            Expr::Eq(eq) => self.visit_eq(&eq),
            Expr::Ne(ne) => self.visit_ne(&ne),
            Expr::Lt(lt) => self.visit_lt(&lt),
            Expr::Le(le) => self.visit_le(&le),
            Expr::Gt(gt) => self.visit_gt(&gt),
            Expr::Ge(ge) => self.visit_ge(&ge),
            Expr::And(and) => self.visit_and(&and),
            Expr::Xor(or) => self.visit_xor(&or),
            Expr::Or(or) => self.visit_or(&or),
            Expr::Not(not) => self.visit_not(&not),
            Expr::Call(call) => self.visit_call(&call),
            Expr::Select(select) => self.visit_select(&select),
            Expr::Alloc(alloc) => self.visit_alloc(&alloc),
            Expr::If(if_) => self.visit_if(&if_),
            Expr::For(for_) => self.visit_for(&for_),
            Expr::While(while_) => self.visit_while(&while_),
            Expr::Let(let_) => self.visit_let(&let_),
            Expr::Function(func) => {
                self.visit_function(&func);
            }
            Expr::Tensor(tensor) => {
                self.visit_tensor(&tensor);
            }
            Expr::None => {}
        }
    }
    fn visit_tensor(&self, _: &Tensor) {}
    fn visit_function(&self, func: &Function) {
        func.body().accept(self);
    }
    fn visit_value(&self, _: &Value) {}
    fn visit_alloc(&self, alloc: &Alloc) {
        alloc.shape().accept(self);
    }
    fn visit_while(&self, while_: &While) {
        while_.cond().accept(self);
        while_.body().accept(self);
    }
    fn visit_variable(&self, _: &Variable) {}
    fn visit_str(&self, _: &Str) {}
    fn visit_cast(&self, cast: &Cast) {
        cast.expr().accept(self);
    }
    fn visit_add(&self, add: &Add) {
        add.lhs().accept(self);
        add.rhs().accept(self);
    }
    fn visit_sub(&self, sub: &Sub) {
        sub.lhs().accept(self);
        sub.rhs().accept(self);
    }
    fn visit_mul(&self, mul: &Mul) {
        mul.lhs().accept(self);
        mul.rhs().accept(self);
    }
    fn visit_div(&self, div: &Div) {
        div.lhs().accept(self);
        div.rhs().accept(self);
    }
    fn visit_mod(&self, mod_: &Mod) {
        mod_.lhs().accept(self);
        mod_.rhs().accept(self);
    }
    fn visit_min(&self, min: &Min) {
        min.lhs().accept(self);
        min.rhs().accept(self);
    }
    fn visit_max(&self, max: &Max) {
        max.lhs().accept(self);
        max.rhs().accept(self);
    }
    fn visit_ge(&self, ge: &Ge) {
        ge.lhs().accept(self);
        ge.rhs().accept(self);
    }
    fn visit_xor(&self, xor: &Xor) {
        xor.lhs().accept(self);
        xor.rhs().accept(self);
    }
    fn visit_gt(&self, gt: &Gt) {
        gt.lhs().accept(self);
        gt.rhs().accept(self);
    }
    fn visit_le(&self, le: &Le) {
        le.lhs().accept(self);
        le.rhs().accept(self);
    }
    fn visit_lt(&self, lt: &Lt) {
        lt.lhs().accept(self);
        lt.rhs().accept(self);
    }
    fn visit_eq(&self, eq: &Eq) {
        eq.lhs().accept(self);
        eq.rhs().accept(self);
    }
    fn visit_ne(&self, ne: &Ne) {
        ne.lhs().accept(self);
        ne.rhs().accept(self);
    }
    fn visit_and(&self, and: &And) {
        and.lhs().accept(self);
        and.rhs().accept(self);
    }
    fn visit_or(&self, or: &Or) {
        or.lhs().accept(self);
        or.rhs().accept(self);
    }
    fn visit_not(&self, not: &Not) {
        not.expr().accept(self);
    }
    fn visit_let(&self, let_: &Let) {
        let_.var().accept(self);
        let_.value().accept(self);
    }
    fn visit_for(&self, for_: &For) {
        for_.var().accept(self);
        for_.start().accept(self);
        for_.end().accept(self);
        for_.body().accept(self);
    }
    fn visit_call(&self, call: &Call) {
        for arg in call.args() {
            arg.accept(self);
        }
    }
    fn visit_select(&self, select: &Select) {
        select.cond().accept(self);
        select.true_value().accept(self);
        select.false_value().accept(self);
    }
    fn visit_if(&self, if_: &If) {
        if_.cond().accept(self);
        if_.then().accept(self);
        if_.else_().accept(self);
    }
}

pub trait HlirMutVisitor where Self: Sized {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Value(value) => self.visit_value(&value),
            Expr::Str(string) => self.visit_str(&string),
            Expr::Variable(var) => self.visit_variable(&var),
            Expr::Cast(cast) => self.visit_cast(&cast),
            Expr::Add(add) => self.visit_add(&add),
            Expr::Sub(sub) => self.visit_sub(&sub),
            Expr::Mul(mul) => self.visit_mul(&mul),
            Expr::Div(div) => self.visit_div(&div),
            Expr::Mod(r#mod) => self.visit_mod(&r#mod),
            Expr::Min(min) => self.visit_min(&min),
            Expr::Max(max) => self.visit_max(&max),
            Expr::Eq(eq) => self.visit_eq(&eq),
            Expr::Ne(ne) => self.visit_ne(&ne),
            Expr::Lt(lt) => self.visit_lt(&lt),
            Expr::Le(le) => self.visit_le(&le),
            Expr::Gt(gt) => self.visit_gt(&gt),
            Expr::Ge(ge) => self.visit_ge(&ge),
            Expr::And(and) => self.visit_and(&and),
            Expr::Xor(or) => self.visit_xor(&or),
            Expr::Or(or) => self.visit_or(&or),
            Expr::Not(not) => self.visit_not(&not),
            Expr::Call(call) => self.visit_call(&call),
            Expr::Select(select) => self.visit_select(&select),
            Expr::Alloc(alloc) => self.visit_alloc(&alloc),
            Expr::If(if_) => self.visit_if(&if_),
            Expr::For(for_) => self.visit_for(&for_),
            Expr::While(while_) => self.visit_while(&while_),
            Expr::Let(let_) => self.visit_let(&let_),
            Expr::Function(func) => {
                self.visit_function(&func);
            }
            Expr::Tensor(tensor) => {
                self.visit_tensor(&tensor);
            }
            Expr::None => {}
        }
    }
    fn visit_tensor(&mut self, _: &Tensor) {}
    fn visit_function(&mut self, func: &Function) {
        func.body().accept_mut(self);
    }
    fn visit_value(&mut self, _: &Value) {}
    fn visit_alloc(&mut self, alloc: &Alloc) {
        alloc.shape().accept_mut(self);
    }
    fn visit_while(&mut self, while_: &While) {
        while_.cond().accept_mut(self);
        while_.body().accept_mut(self);
    }
    fn visit_variable(&mut self, _: &Variable) {}
    fn visit_str(&mut self, _: &Str) {}
    fn visit_cast(&mut self, cast: &Cast) {
        cast.expr().accept_mut(self);
    }
    fn visit_add(&mut self, add: &Add) {
        add.lhs().accept_mut(self);
        add.rhs().accept_mut(self);
    }
    fn visit_sub(&mut self, sub: &Sub) {
        sub.lhs().accept_mut(self);
        sub.rhs().accept_mut(self);
    }
    fn visit_mul(&mut self, mul: &Mul) {
        mul.lhs().accept_mut(self);
        mul.rhs().accept_mut(self);
    }
    fn visit_div(&mut self, div: &Div) {
        div.lhs().accept_mut(self);
        div.rhs().accept_mut(self);
    }
    fn visit_mod(&mut self, mod_: &Mod) {
        mod_.lhs().accept_mut(self);
        mod_.rhs().accept_mut(self);
    }
    fn visit_min(&mut self, min: &Min) {
        min.lhs().accept_mut(self);
        min.rhs().accept_mut(self);
    }
    fn visit_max(&mut self, max: &Max) {
        max.lhs().accept_mut(self);
        max.rhs().accept_mut(self);
    }
    fn visit_ge(&mut self, ge: &Ge) {
        ge.lhs().accept_mut(self);
        ge.rhs().accept_mut(self);
    }
    fn visit_xor(&mut self, xor: &Xor) {
        xor.lhs().accept_mut(self);
        xor.rhs().accept_mut(self);
    }
    fn visit_gt(&mut self, gt: &Gt) {
        gt.lhs().accept_mut(self);
        gt.rhs().accept_mut(self);
    }
    fn visit_le(&mut self, le: &Le) {
        le.lhs().accept_mut(self);
        le.rhs().accept_mut(self);
    }
    fn visit_lt(&mut self, lt: &Lt) {
        lt.lhs().accept_mut(self);
        lt.rhs().accept_mut(self);
    }
    fn visit_eq(&mut self, eq: &Eq) {
        eq.lhs().accept_mut(self);
        eq.rhs().accept_mut(self);
    }
    fn visit_ne(&mut self, ne: &Ne) {
        ne.lhs().accept_mut(self);
        ne.rhs().accept_mut(self);
    }
    fn visit_and(&mut self, and: &And) {
        and.lhs().accept_mut(self);
        and.rhs().accept_mut(self);
    }
    fn visit_or(&mut self, or: &Or) {
        or.lhs().accept_mut(self);
        or.rhs().accept_mut(self);
    }
    fn visit_not(&mut self, not: &Not) {
        not.expr().accept_mut(self);
    }
    fn visit_let(&mut self, let_: &Let) {
        let_.var().accept_mut(self);
        let_.value().accept_mut(self);
    }
    fn visit_for(&mut self, for_: &For) {
        for_.var().accept_mut(self);
        for_.start().accept_mut(self);
        for_.end().accept_mut(self);
        for_.body().accept_mut(self);
    }
    fn visit_call(&mut self, call: &Call) {
        for arg in call.args() {
            arg.accept_mut(self);
        }
    }
    fn visit_select(&mut self, select: &Select) {
        select.cond().accept_mut(self);
        select.true_value().accept_mut(self);
        select.false_value().accept_mut(self);
    }
    fn visit_if(&mut self, if_: &If) {
        if_.cond().accept_mut(self);
        if_.then().accept_mut(self);
        if_.else_().accept_mut(self);
    }
}

pub trait HlirMutateVisitor where Self: Sized {
    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Value(value) => self.visit_value(value),
            Expr::Str(string) => self.visit_str(string),
            Expr::Variable(var) => self.visit_variable(var),
            Expr::Cast(cast) => self.visit_cast(cast),
            Expr::Add(add) => self.visit_add(add),
            Expr::Sub(sub) => self.visit_sub(sub),
            Expr::Mul(mul) => self.visit_mul(mul),
            Expr::Div(div) => self.visit_div(div),
            Expr::Mod(r#mod) => self.visit_mod(r#mod),
            Expr::Min(min) => self.visit_min(min),
            Expr::Max(max) => self.visit_max(max),
            Expr::Eq(eq) => self.visit_eq(eq),
            Expr::Ne(ne) => self.visit_ne(ne),
            Expr::Lt(lt) => self.visit_lt(lt),
            Expr::Le(le) => self.visit_le(le),
            Expr::Gt(gt) => self.visit_gt(gt),
            Expr::Ge(ge) => self.visit_ge(ge),
            Expr::And(and) => self.visit_and(and),
            Expr::Xor(or) => self.visit_xor(or),
            Expr::Or(or) => self.visit_or(or),
            Expr::Not(not) => self.visit_not(not),
            Expr::Call(call) => self.visit_call(call),
            Expr::Select(select) => self.visit_select(select),
            Expr::Alloc(alloc) => self.visit_alloc(alloc),
            Expr::If(if_) => self.visit_if(if_),
            Expr::For(for_) => self.visit_for(for_),
            Expr::While(while_) => self.visit_while(while_),
            Expr::Let(let_) => self.visit_let(let_),
            Expr::Function(func) => {
                self.visit_function(func);
            }
            Expr::Tensor(tensor) => {
                self.visit_tensor(tensor);
            }
            Expr::None => {}
        }
    }
    fn visit_tensor(&mut self, _: &Tensor) {}
    fn visit_function(&mut self, func: &Function) {
        func.body().accept_mutate(self);
    }
    fn visit_value(&mut self, _: &Value) {}
    fn visit_alloc(&mut self, alloc: &Alloc) {
        alloc.shape().accept_mutate(self);
    }
    fn visit_while(&mut self, while_: &While) {
        while_.cond().accept_mutate(self);
        while_.body().accept_mutate(self);
    }
    fn visit_variable(&mut self, _: &Variable) {}
    fn visit_str(&mut self, _: &Str) {}
    fn visit_cast(&mut self, cast: &Cast) {
        cast.expr().accept_mutate(self);
    }
    fn visit_add(&mut self, add: &Add) {
        add.lhs().accept_mutate(self);
        add.rhs().accept_mutate(self);
    }
    fn visit_sub(&mut self, sub: &Sub) {
        sub.lhs().accept_mutate(self);
        sub.rhs().accept_mutate(self);
    }
    fn visit_mul(&mut self, mul: &Mul) {
        mul.lhs().accept_mutate(self);
        mul.rhs().accept_mutate(self);
    }
    fn visit_div(&mut self, div: &Div) {
        div.lhs().accept_mutate(self);
        div.rhs().accept_mutate(self);
    }
    fn visit_mod(&mut self, mod_: &Mod) {
        mod_.lhs().accept_mutate(self);
        mod_.rhs().accept_mutate(self);
    }
    fn visit_min(&mut self, min: &Min) {
        min.lhs().accept_mutate(self);
        min.rhs().accept_mutate(self);
    }
    fn visit_max(&mut self, max: &Max) {
        max.lhs().accept_mutate(self);
        max.rhs().accept_mutate(self);
    }
    fn visit_ge(&mut self, ge: &Ge) {
        ge.lhs().accept_mutate(self);
        ge.rhs().accept_mutate(self);
    }
    fn visit_xor(&mut self, xor: &Xor) {
        xor.lhs().accept_mutate(self);
        xor.rhs().accept_mutate(self);
    }
    fn visit_gt(&mut self, gt: &Gt) {
        gt.lhs().accept_mutate(self);
        gt.rhs().accept_mutate(self);
    }
    fn visit_le(&mut self, le: &Le) {
        le.lhs().accept_mutate(self);
        le.rhs().accept_mutate(self);
    }
    fn visit_lt(&mut self, lt: &Lt) {
        lt.lhs().accept_mutate(self);
        lt.rhs().accept_mutate(self);
    }
    fn visit_eq(&mut self, eq: &Eq) {
        eq.lhs().accept_mutate(self);
        eq.rhs().accept_mutate(self);
    }
    fn visit_ne(&mut self, ne: &Ne) {
        ne.lhs().accept_mutate(self);
        ne.rhs().accept_mutate(self);
    }
    fn visit_and(&mut self, and: &And) {
        and.lhs().accept_mutate(self);
        and.rhs().accept_mutate(self);
    }
    fn visit_or(&mut self, or: &Or) {
        or.lhs().accept_mutate(self);
        or.rhs().accept_mutate(self);
    }
    fn visit_not(&mut self, not: &Not) {
        not.expr().accept_mutate(self);
    }
    fn visit_let(&mut self, let_: &Let) {
        let_.var().accept_mutate(self);
        let_.value().accept_mutate(self);
    }
    fn visit_for(&mut self, for_: &For) {
        for_.var().accept_mutate(self);
        for_.start().accept_mutate(self);
        for_.end().accept_mutate(self);
        for_.body().accept_mutate(self);
    }
    fn visit_call(&mut self, call: &Call) {
        for arg in call.args() {
            arg.accept_mutate(self);
        }
    }
    fn visit_select(&mut self, select: &Select) {
        select.cond().accept_mutate(self);
        select.true_value().accept_mutate(self);
        select.false_value().accept_mutate(self);
    }
    fn visit_if(&mut self, if_: &If) {
        if_.cond().accept_mutate(self);
        if_.then().accept_mutate(self);
        if_.else_().accept_mutate(self);
    }
}
