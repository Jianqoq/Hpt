use crate::halide::variable::Variable;

use super::{ exprs::*, expr::Expr };

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
            Expr::Tuple(tuple) => {
                self.visit_tuple(&tuple);
            }
            Expr::Type(a) => {
                match a {
                    super::func_type::Type::Dtype(_) => {}
                    super::func_type::Type::Tensor(tensor_type) => {
                        self.visit_tensor_type(tensor_type);
                    }
                    super::func_type::Type::Tuple(tuple) => {
                        self.visit_tuple(tuple);
                    }
                    super::func_type::Type::Ptr(_) => {}
                    super::func_type::Type::Str => {}
                    super::func_type::Type::None => {}
                }
            }
            Expr::TensorType(a) => {
                self.visit_tensor_type(a);
            }
            Expr::Slice(slcie) => {
                self.visit_slice(slcie);
            }
            Expr::OpNode(op_node) => {
                self.visit_op_node(op_node);
            }
            Expr::Tensor(cmp_node) => {
                self.visit_tensor(cmp_node);
            }
            Expr::Return(return_) => {
                self.visit_return(&return_);
            }
            Expr::None => {}
        }
    }
    fn visit_return(&self, ret: &Return) {
        for value in ret.expr() {
            value.accept(self);
        }
    }
    fn visit_tensor(&self, _: &Tensor) {}
    fn visit_op_node(&self, _: &OpNode) {}
    fn visit_slice(&self, slice: &Slice) {
        slice.var().accept(self);
        for (start, end, step) in slice.selections() {
            start.accept(self);
            end.accept(self);
            step.accept(self);
        }
    }
    fn visit_tensor_type(&self, _: &TensorType) {}
    fn visit_tuple(&self, tuple: &Tuple) {
        for value in tuple.values() {
            value.accept(self);
        }
    }
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

pub trait MutatorGetSet {
    fn set_expr<T: Into<Expr>>(&mut self, expr: T);
    fn expr(&self) -> &Expr;
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
            Expr::Return(return_) => self.visit_return(&return_),
            Expr::Function(func) => {
                self.visit_function(&func);
            }
            Expr::Tuple(tuple) => {
                self.visit_tuple(&tuple);
            }
            Expr::Type(a) => {
                match a {
                    super::func_type::Type::Dtype(_) => {}
                    super::func_type::Type::Tensor(tensor_type) => {
                        self.visit_tensor_type(tensor_type);
                    }
                    super::func_type::Type::Tuple(tuple) => {
                        self.visit_tuple(tuple);
                    }
                    super::func_type::Type::Ptr(_) => {}
                    super::func_type::Type::Str => {}
                    super::func_type::Type::None => {}
                }
            }
            Expr::TensorType(a) => {
                self.visit_tensor_type(a);
            }
            Expr::Slice(slcie) => {
                self.visit_slice(slcie);
            }
            Expr::OpNode(op_node) => {
                self.visit_op_node(op_node);
            }
            Expr::Tensor(cmp_node) => {
                self.visit_tensor(cmp_node);
            }
            Expr::None => {}
        }
    }
    fn visit_return(&mut self, ret: &Return) {
        for value in ret.expr() {
            value.accept_mut(self);
        }
    }
    fn visit_tensor(&mut self, _: &Tensor) {}
    fn visit_op_node(&mut self, _: &OpNode) {}
    fn visit_slice(&mut self, slice: &Slice) {
        slice.var().accept_mut(self);
        for (start, end, step) in slice.selections() {
            start.accept_mut(self);
            end.accept_mut(self);
            step.accept_mut(self);
        }
    }
    fn visit_tensor_type(&mut self, _: &TensorType) {}
    fn visit_tuple(&mut self, tuple: &Tuple) {
        for value in tuple.values() {
            value.accept_mut(self);
        }
    }
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

pub(crate) fn mutate_expr<V, T: Into<Expr>>(visitor: &mut V, expr: T) -> Expr
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let expr = expr.into();
    if expr.is_none() {
        visitor.set_expr(Expr::None);
    } else {
        expr.accept_mutate(visitor);
    }
    return visitor.expr().clone();
}

pub(crate) fn visit_variable<V>(visitor: &mut V, var: &Variable)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    visitor.set_expr(var);
}

pub(crate) fn visit_str<V>(visitor: &mut V, string: &Str)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    visitor.set_expr(string);
}

pub(crate) fn visit_cast<V>(visitor: &mut V, cast: &Cast)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let a = mutate_expr(visitor, cast.expr());
    if &a == cast.expr() {
        visitor.set_expr(cast);
    } else {
        visitor.set_expr(Cast::make(a, cast.dtype()));
    }
}

macro_rules! mutate_binop {
    ($fn_name:ident, $op:ident, $T:ident) => {
        pub(crate) fn $fn_name <V>(visitor: &mut V, $op: &$T)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
        {
            let a = visitor.mutate_expr($op.lhs());
            let b = visitor.mutate_expr($op.rhs());
            if &a == $op.lhs() && &b == $op.rhs() {
                visitor.set_expr($op);
            } else {
                visitor.set_expr($T::make(a, b));
            }
        }
    };
}

mutate_binop!(visit_add, add, Add);
mutate_binop!(visit_sub, sub, Sub);
mutate_binop!(visit_mul, mul, Mul);
mutate_binop!(visit_div, div, Div);
mutate_binop!(visit_mod, mod_, Mod);
mutate_binop!(visit_min, min, Min);
mutate_binop!(visit_max, max, Max);
mutate_binop!(visit_ge, ge, Ge);
mutate_binop!(visit_xor, xor, Xor);
mutate_binop!(visit_gt, gt, Gt);
mutate_binop!(visit_le, le, Le);
mutate_binop!(visit_lt, lt, Lt);
mutate_binop!(visit_eq, eq, Eq);
mutate_binop!(visit_ne, ne, Ne);
mutate_binop!(visit_and, and, And);
mutate_binop!(visit_or, or, Or);

pub(crate) fn visit_not<V>(visitor: &mut V, not: &Not)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let a = mutate_expr(visitor, not.expr());
    if &a == not.expr() {
        visitor.set_expr(not);
    } else {
        visitor.set_expr(Not::make(a));
    }
}

pub(crate) fn visit_let<V>(visitor: &mut V, let_stmt: &Let)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let name = let_stmt.var().into();
    let e1 = let_stmt.value();
    let body = let_stmt.body();
    let var = mutate_expr(visitor, &name);
    let val = mutate_expr(visitor, e1);
    let new_body = mutate_expr(visitor, body);
    if &var == &name && &val == e1 && &new_body == body {
        visitor.set_expr(let_stmt);
    } else {
        if let Some(var) = var.to_variable() {
            visitor.set_expr(Let::make_from_expr(var.clone(), val, new_body));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", name, var);
            visitor.set_expr(Expr::None);
        }
    }
}

pub(crate) fn visit_for<V>(visitor: &mut V, for_: &For)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let var = for_.var().into();
    let new_var = visitor.mutate_expr(&var);
    let start = visitor.mutate_expr(for_.start());
    let end = visitor.mutate_expr(for_.end());
    let step = visitor.mutate_expr(for_.step());
    let body = visitor.mutate_expr(for_.body());
    if
        &new_var == &var &&
        &start == for_.start() &&
        &end == for_.end() &&
        &step == for_.step() &&
        &body == for_.body()
    {
        visitor.set_expr(for_);
    } else {
        if let Some(new_var) = new_var.to_variable() {
            visitor.set_expr(For::make(new_var.clone(), start, end, step, body));
        } else {
            eprintln!("Failed to convert variable, from: {} to: {}", var, new_var);
            visitor.set_expr(Expr::None);
        }
    }
}

pub(crate) fn visit_call<V>(visitor: &mut V, call: &Call)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let args = call.args();
    let mut changed = false;
    let mut new_args = Vec::with_capacity(args.len());
    for arg in args.iter() {
        let new_arg = visitor.mutate_expr(arg);
        if &new_arg != arg {
            changed = true;
        }
        new_args.push(new_arg);
    }
    if !changed {
        visitor.set_expr(call);
    } else {
        visitor.set_expr(Call::make(call.name(), &new_args));
    }
}

pub(crate) fn visit_select<V>(visitor: &mut V, select: &Select)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let cond = visitor.mutate_expr(select.cond());
    let true_value = visitor.mutate_expr(select.true_value());
    let false_value = visitor.mutate_expr(select.false_value());
    if
        &cond == select.cond() &&
        &true_value == select.true_value() &&
        &false_value == select.false_value()
    {
        visitor.set_expr(select);
    } else {
        visitor.set_expr(Select::make(cond, true_value, false_value));
    }
}

pub(crate) fn visit_if<V>(visitor: &mut V, if_then_else: &If)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let cond = visitor.mutate_expr(if_then_else.cond());
    let then_case = visitor.mutate_expr(if_then_else.then());
    let else_case = visitor.mutate_expr(if_then_else.else_());
    if
        &cond == if_then_else.cond() &&
        &then_case == if_then_else.then() &&
        &else_case == if_then_else.else_()
    {
        visitor.set_expr(if_then_else);
    } else {
        visitor.set_expr(If::make(cond, then_case, else_case));
    }
}

pub(crate) fn visit_alloc<V>(visitor: &mut V, alloc: &Alloc)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let shape = visitor.mutate_expr(alloc.shape());
    if &shape == alloc.shape() {
        visitor.set_expr(alloc);
    } else {
        visitor.set_expr(Alloc::make(shape, alloc.dtype()));
    }
}

pub(crate) fn visit_while<V>(visitor: &mut V, while_: &While)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let cond = visitor.mutate_expr(while_.cond());
    let body = visitor.mutate_expr(while_.body());
    if &cond == while_.cond() && &body == while_.body() {
        visitor.set_expr(while_);
    } else {
        visitor.set_expr(While::make(cond, body));
    }
}

pub(crate) fn visit_function<V>(visitor: &mut V, func: &Function)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let body = visitor.mutate_expr(func.body());
    let return_type = visitor.mutate_expr(func.return_type());
    if &body == func.body() && &return_type == func.return_type() {
        visitor.set_expr(func);
    } else {
        visitor.set_expr(
            Function::make(func.name(), func.args(), return_type.to_type().unwrap(), body)
        );
    }
}

pub(crate) fn visit_tuple<V>(visitor: &mut V, tuple: &Tuple)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let mut changed = false;
    let mut new_values = Vec::with_capacity(tuple.values().len());
    for value in tuple.values() {
        let new_value = visitor.mutate_expr(value);
        if &new_value != value {
            changed = true;
        }
        new_values.push(new_value);
    }
    if !changed {
        visitor.set_expr(tuple);
    } else {
        visitor.set_expr(Tuple::make(&new_values));
    }
}

pub(crate) fn visit_slcie<V>(visitor: &mut V, slice: &Slice)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let var = visitor.mutate_expr(slice.var());
    let mut changed = false;
    let mut new_selections = Vec::with_capacity(slice.selections().len());
    for (start, end, step) in slice.selections() {
        let new_start = visitor.mutate_expr(start);
        let new_end = visitor.mutate_expr(end);
        let new_step = visitor.mutate_expr(step);
        if &new_start != start || &new_end != end || &new_step != step {
            changed = true;
        }
        new_selections.push((new_start, new_end, new_step));
    }
    if !changed {
        visitor.set_expr(slice);
    } else {
        visitor.set_expr(Slice::make(var.to_variable().unwrap().clone(), new_selections));
    }
}

pub(crate) fn visit_return<V>(visitor: &mut V, return_: &Return)
    where V: MutatorGetSet + Sized + HlirMutateVisitor
{
    let mut changed = false;
    let mut new_exprs = Vec::with_capacity(return_.expr().len());
    for expr in return_.expr() {
        let new_expr = visitor.mutate_expr(expr);
        if &new_expr != expr {
            changed = true;
        }
        new_exprs.push(new_expr);
    }
    if !changed {
        visitor.set_expr(return_);
    } else {
        visitor.set_expr(Return::make(&new_exprs));
    }
}

pub trait HlirMutateVisitor where Self: Sized + MutatorGetSet {
    fn mutate_expr<T: Into<Expr>>(&mut self, expr: T) -> Expr {
        mutate_expr(self, expr)
    }

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
            Expr::Tuple(tuple) => {
                self.visit_tuple(tuple);
            }
            Expr::Type(a) => {
                match a {
                    super::func_type::Type::Dtype(_) => {}
                    super::func_type::Type::Tensor(tensor_type) => {
                        self.set_expr(tensor_type.clone());
                    }
                    super::func_type::Type::Tuple(tuple) => {
                        self.set_expr(tuple.clone());
                    }
                    super::func_type::Type::Ptr(_) => {}
                    super::func_type::Type::Str => todo!(),
                    super::func_type::Type::None => {}
                }
            }
            Expr::TensorType(a) => {
                self.set_expr(a.clone());
            }
            Expr::Slice(slcie) => {
                self.visit_slice(slcie);
            }
            Expr::OpNode(op_node) => {
                self.visit_op_node(op_node);
            }
            Expr::Tensor(cmp_node) => {
                self.visit_tensor(cmp_node);
            }
            Expr::Return(return_) => {
                self.visit_return(return_);
            }
            Expr::None => {}
        }
    }
    fn visit_return(&mut self, ret: &Return) {
        visit_return(self, ret);
    }
    fn visit_tensor(&mut self, _: &Tensor) {}
    fn visit_op_node(&mut self, _: &OpNode) {}
    fn visit_slice(&mut self, slice: &Slice) {
        visit_slcie(self, slice);
    }
    fn visit_tuple(&mut self, tuple: &Tuple) {
        visit_tuple(self, tuple)
    }
    fn visit_function(&mut self, func: &Function) {
        visit_function(self, func)
    }
    fn visit_value(&mut self, val: &Value) {
        self.set_expr(val.clone());
    }
    fn visit_alloc(&mut self, alloc: &Alloc) {
        visit_alloc(self, alloc);
    }
    fn visit_while(&mut self, while_: &While) {
        visit_while(self, while_)
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
    fn visit_xor(&mut self, xor: &Xor) {
        visit_xor(self, xor);
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
    fn visit_or(&mut self, or: &Or) {
        visit_or(self, or);
    }
    fn visit_not(&mut self, not: &Not) {
        visit_not(self, not);
    }
    fn visit_let(&mut self, let_: &Let) {
        visit_let(self, let_);
    }
    fn visit_for(&mut self, for_: &For) {
        visit_for(self, for_);
    }
    fn visit_call(&mut self, call: &Call) {
        visit_call(self, call);
    }
    fn visit_select(&mut self, select: &Select) {
        visit_select(self, select);
    }
    fn visit_if(&mut self, if_: &If) {
        visit_if(self, if_);
    }
}

pub trait IntoVar {
    fn into_var(self) -> Variable;
}
