#![allow(unused)]

use std::hash::{ BuildHasher, Hash, Hasher };

use super::{
    prime_expr::PrimeExpr,
    exprs::{ Cast, Float, Int, Str, UInt },
    r#type::Type,
    stmt::Stmt,
    traits::{ AccepterMut, IRMutVisitor },
    variable::Variable,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpResult {
    Equal,
    LessThan,
    GreaterThan,
    Unknown,
}

struct IRComparator {
    result: CmpResult,
    expr_: PrimeExpr,
    stmt_: Stmt,
}

impl IRComparator {
    pub fn new() -> Self {
        IRComparator {
            result: CmpResult::Equal,
            expr_: PrimeExpr::None,
            stmt_: Stmt::None,
        }
    }

    pub fn compare_scalar<T: PartialEq + PartialOrd>(&mut self, a: &T, b: &T) -> CmpResult {
        if self.result != CmpResult::Equal {
            return self.result;
        }
        if a < b {
            self.result = CmpResult::LessThan;
        } else if a > b {
            self.result = CmpResult::GreaterThan;
        }
        return self.result;
    }

    pub fn compare_names(&mut self, a: &str, b: &str) -> CmpResult {
        if self.result != CmpResult::Equal {
            return self.result;
        }
        if a < b {
            self.result = CmpResult::LessThan;
        } else if a > b {
            self.result = CmpResult::GreaterThan;
        }
        return self.result;
    }

    pub fn compare_expr<T: Into<PrimeExpr>>(&mut self, a: T, b: T) -> CmpResult {
        if self.result != CmpResult::Equal {
            return self.result;
        }
        let a: PrimeExpr = a.into();
        let b: PrimeExpr = b.into();
        if &a == &b {
            return CmpResult::Equal;
        }
        if a.is_none() {
            return CmpResult::LessThan;
        }
        if b.is_none() {
            return CmpResult::GreaterThan;
        }

        let mut a_hash = hashbrown::hash_map::DefaultHashBuilder::default().build_hasher();
        let mut b_hash = hashbrown::hash_map::DefaultHashBuilder::default().build_hasher();
        a.hash(&mut a_hash);
        b.hash(&mut b_hash);
        let a_hash_value = a_hash.finish();
        let b_hash_value = b_hash.finish();
        if self.compare_scalar(&a_hash_value, &b_hash_value) != CmpResult::Equal {
            return self.result;
        }

        if self.compare_scalar(&a.type_info(), &b.type_info()) != CmpResult::Equal {
            return self.result;
        }

        return self.result;
    }

    pub fn compare_exprs<T: Into<PrimeExpr>>(&mut self, a: &[T], b: &[T]) -> CmpResult
        where for<'a> &'a T: Into<PrimeExpr>
    {
        if self.result != CmpResult::Equal {
            return self.result;
        }
        if a.len() < b.len() {
            self.result = CmpResult::LessThan;
        } else if a.len() > b.len() {
            self.result = CmpResult::GreaterThan;
        } else {
            for (a, b) in a.iter().zip(b.iter()) {
                self.compare_expr(a, b);
                if self.result != CmpResult::Equal {
                    break;
                }
            }
        }
        return self.result;
    }

    pub fn compare_stmt<T: Into<Stmt>>(&mut self, a: T, b: T) -> CmpResult {
        let a: Stmt = a.into();
        let b: Stmt = b.into();
        if self.result != CmpResult::Equal {
            return self.result;
        }
        if a.same_as(&b) {
            return CmpResult::Equal;
        }

        if a.is_none() {
            return CmpResult::LessThan;
        }
        if b.is_none() {
            return CmpResult::GreaterThan;
        }

        if self.compare_scalar(&a.type_info(), &b.type_info()) != CmpResult::Equal {
            return self.result;
        }

        self.stmt_ = a;
        b.accept_mut(self);

        return self.result;
    }

    pub fn compare_types(&mut self, a: Type, b: Type) -> CmpResult {
        if self.result != CmpResult::Equal {
            return self.result;
        }
        self.compare_scalar(&a.code(), &b.code());
        self.compare_scalar(&a.bits(), &b.bits());
        self.compare_scalar(&a.lanes(), &b.lanes())
    }
}

impl IRMutVisitor for IRComparator {
    fn visit_int(&mut self, int: &Int) {
        let expr = self.expr_.to_int();
        if let Some(int) = expr {
            self.compare_scalar(&int.value(), &int.value());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_uint(&mut self, uint: &UInt) {
        let expr = self.expr_.to_uint();
        if let Some(uint) = expr {
            self.compare_scalar(&uint.value(), &uint.value());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_float(&mut self, float: &Float) {
        let expr = self.expr_.to_float();
        if let Some(float) = expr {
            self.compare_scalar(&float.value(), &float.value());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_str(&mut self, string: &Str) {
        let expr = self.expr_.to_str().cloned();
        if let Some(string) = expr {
            self.compare_names(string.value(), string.value());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_cast(&mut self, cast: &Cast) {
        let expr = self.expr_.to_cast().cloned();
        if let Some(cast) = expr {
            self.compare_expr(cast.expr(), cast.expr());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_variable(&mut self, var: &Variable) {
        let expr = self.expr_.to_variable().cloned();
        if let Some(var) = expr {
            self.compare_names(var.name(), var.name());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_add(&mut self, add: &super::exprs::Add) {
        let expr = self.expr_.to_add().cloned();
        if let Some(add) = expr {
            self.compare_expr(add.e1(), add.e1());
            self.compare_expr(add.e2(), add.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_and(&mut self, and: &super::exprs::And) {
        let expr = self.expr_.to_and().cloned();
        if let Some(and) = expr {
            self.compare_expr(and.e1(), and.e1());
            self.compare_expr(and.e2(), and.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_div(&mut self, div: &super::exprs::Div) {
        let expr = self.expr_.to_div().cloned();
        if let Some(div) = expr {
            self.compare_expr(div.e1(), div.e1());
            self.compare_expr(div.e2(), div.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_mul(&mut self, mul: &super::exprs::Mul) {
        let expr = self.expr_.to_mul().cloned();
        if let Some(mul) = expr {
            self.compare_expr(mul.e1(), mul.e1());
            self.compare_expr(mul.e2(), mul.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_or(&mut self, or: &super::exprs::Or) {
        let expr = self.expr_.to_or().cloned();
        if let Some(or) = expr {
            self.compare_expr(or.e1(), or.e1());
            self.compare_expr(or.e2(), or.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_sub(&mut self, sub: &super::exprs::Sub) {
        let expr = self.expr_.to_sub().cloned();
        if let Some(sub) = expr {
            self.compare_expr(sub.e1(), sub.e1());
            self.compare_expr(sub.e2(), sub.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_mod(&mut self, mod_: &super::exprs::Mod) {
        let expr = self.expr_.to_mod().cloned();
        if let Some(mod_) = expr {
            self.compare_expr(mod_.e1(), mod_.e1());
            self.compare_expr(mod_.e2(), mod_.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_eq(&mut self, eq: &super::exprs::Eq) {
        let expr = self.expr_.to_eq().cloned();
        if let Some(eq) = expr {
            self.compare_expr(eq.e1(), eq.e1());
            self.compare_expr(eq.e2(), eq.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_ne(&mut self, ne: &super::exprs::Ne) {
        let expr = self.expr_.to_ne().cloned();
        if let Some(ne) = expr {
            self.compare_expr(ne.e1(), ne.e1());
            self.compare_expr(ne.e2(), ne.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_lt(&mut self, lt: &super::exprs::Lt) {
        let expr = self.expr_.to_lt().cloned();
        if let Some(lt) = expr {
            self.compare_expr(lt.e1(), lt.e1());
            self.compare_expr(lt.e2(), lt.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_le(&mut self, le: &super::exprs::Le) {
        let expr = self.expr_.to_le().cloned();
        if let Some(le) = expr {
            self.compare_expr(le.e1(), le.e1());
            self.compare_expr(le.e2(), le.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_gt(&mut self, gt: &super::exprs::Gt) {
        let expr = self.expr_.to_gt().cloned();
        if let Some(gt) = expr {
            self.compare_expr(gt.e1(), gt.e1());
            self.compare_expr(gt.e2(), gt.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_ge(&mut self, ge: &super::exprs::Ge) {
        let expr = self.expr_.to_ge().cloned();
        if let Some(ge) = expr {
            self.compare_expr(ge.e1(), ge.e1());
            self.compare_expr(ge.e2(), ge.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_not(&mut self, not: &super::exprs::Not) {
        let expr = self.expr_.to_not().cloned();
        if let Some(not) = expr {
            self.compare_expr(not.e(), not.e());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_min(&mut self, min: &super::exprs::Min) {
        let expr = self.expr_.to_min().cloned();
        if let Some(min) = expr {
            self.compare_expr(min.e1(), min.e1());
            self.compare_expr(min.e2(), min.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_max(&mut self, max: &super::exprs::Max) {
        let expr = self.expr_.to_max().cloned();
        if let Some(max) = expr {
            self.compare_expr(max.e1(), max.e1());
            self.compare_expr(max.e2(), max.e2());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_load(&mut self, load: &super::exprs::Load) {
        let expr = self.expr_.to_load().cloned();
        if let Some(load) = expr {
            self.compare_expr(load.name(), load.name());
            for (a, b) in load.indices().iter().zip(load.indices().iter()) {
                self.compare_expr(a, b);
            }
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_call(&mut self, call: &super::exprs::Call) {
        let expr = self.expr_.to_call().cloned();
        if let Some(expr) = expr {
            self.compare_expr(expr.name(), call.name());
            let self_expr = expr
                .args()
                .iter()
                .map(|x| {
                    let res = x.as_ref();
                    res
                })
                .collect::<Vec<&PrimeExpr>>();
            let call_expr = call
                .args()
                .iter()
                .map(|x| {
                    let res = x.as_ref();
                    res
                })
                .collect::<Vec<&PrimeExpr>>();
            self.compare_exprs(&self_expr, &call_expr);
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_let_stmt(&mut self, let_stmt: &super::let_stmt::LetStmt) {
        let expr = self.stmt_.to_let_stmt().cloned();
        if let Some(let_stmt) = expr {
            self.compare_expr(let_stmt.var(), let_stmt.var());
            self.compare_expr(let_stmt.body(), let_stmt.body());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_let(&mut self, let_stmt: &super::exprs::Let) {
        let expr = self.expr_.to_let().cloned();
        if let Some(let_stmt) = expr {
            self.compare_expr(let_stmt.name(), let_stmt.name());
            self.compare_expr(let_stmt.e1(), let_stmt.e1());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_for(&mut self, for_stmt: &super::for_stmt::For) {
        let expr = self.stmt_.to_for_stmt().cloned();
        if let Some(for_stmt) = expr {
            self.compare_expr(for_stmt.var(), for_stmt.var());
            self.compare_expr(for_stmt.start(), for_stmt.start());
            self.compare_expr(for_stmt.end(), for_stmt.end());
            self.compare_stmt(for_stmt.stmt(), for_stmt.stmt());
        } else {
            self.result = CmpResult::LessThan;
        }
    }

    fn visit_store(&mut self, store: &super::store_stmt::StoreStmt) {
        let expr = self.stmt_.to_store_stmt().cloned();
        if let Some(store) = expr {
            self.compare_expr(store.var(), store.var());
            self.compare_expr(store.indices(), store.indices());
            self.compare_expr(store.val(), store.val());
        } else {
            self.result = CmpResult::LessThan;
        }
    }
}

pub fn stmt_equal(a: &Stmt, b: &Stmt) -> bool {
    let mut comparator = IRComparator::new();
    comparator.compare_stmt(a, b);
    comparator.result == CmpResult::Equal
}

pub fn expr_equal<T: Into<PrimeExpr>>(a: T, b: T) -> bool {
    let mut comparator = IRComparator::new();
    comparator.compare_expr(a, b);
    comparator.result == CmpResult::Equal
}
