use tensor_types::type_promote::{ BitWiseOut, FloatOut, NormalOut };

use crate::halide::{
    exprs::{ Add, Float, Int },
    prime_expr::PrimeExpr,
    stmt::Stmt,
    traits::{ IRMutateVisitor, MutatorGetSet },
};

pub struct ConstFold {
    pub(crate) folded: bool,
    pub(crate) expr: PrimeExpr,
    pub(crate) stmt: Stmt,
}

impl ConstFold {
    pub fn new() -> Self {
        Self {
            folded: false,
            expr: PrimeExpr::None,
            stmt: Stmt::None,
        }
    }

    pub fn const_fold(&mut self, mut expr: PrimeExpr) -> PrimeExpr {
        self.folded = false;
        while self.folded {
            self.expr = PrimeExpr::None;
            self.stmt = Stmt::None;
            self.folded = false;
            expr = self.mutate_expr(&expr);
        }
        self.expr.clone()
    }
}

impl MutatorGetSet for ConstFold {
    fn set_expr<T: Into<PrimeExpr>>(&mut self, expr: T) {
        self.expr = expr.into();
    }

    fn set_stmt<T: Into<Stmt>>(&mut self, stmt: T) {
        self.stmt = stmt.into();
    }

    fn expr(&self) -> &PrimeExpr {
        &self.expr
    }

    fn stmt(&self) -> &Stmt {
        &self.stmt
    }
}

impl IRMutateVisitor for ConstFold {
    fn visit_add(&mut self, add: &crate::halide::exprs::Add) {
        let e1 = self.mutate_expr(add.e1());
        let e2 = self.mutate_expr(add.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(Int::make(i1.dtype()._add(*i2.dtype()), i1.value() + i2.value()))
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(i1.dtype()._add(*f2.dtype()), (i1.value() as f64) + f2.value())
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._add(*i2.dtype()), f1.value() + (i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), _) => {
                if i1.value() == 0 {
                    self.set_expr(e2);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Add(Add::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Int(i2)) => {
                if i2.value() == 0 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Add(Add::make(e1, e2)));
                }
            }
            (PrimeExpr::Float(f1), _) => {
                if f1.value() == 0.0 {
                    self.set_expr(e2);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Add(Add::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Float(f2)) => {
                if f2.value() == 0.0 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Add(Add::make(e1, e2)));
                }
            }
            _ => {
                self.set_expr(PrimeExpr::Add(Add::make(e1, e2)));
            }
        }
    }
    fn visit_sub(&mut self, sub: &crate::halide::exprs::Sub) {
        let e1 = self.mutate_expr(sub.e1());
        let e2 = self.mutate_expr(sub.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(Int::make(i1.dtype()._sub(*i2.dtype()), i1.value() - i2.value()))
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(i1.dtype()._sub(*f2.dtype()), (i1.value() as f64) - f2.value())
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._sub(*i2.dtype()), f1.value() - (i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), _) => {
                if i1.value() == 0 {
                    self.set_expr(PrimeExpr::Neg(crate::halide::exprs::Neg::make(e2)));
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Sub(crate::halide::exprs::Sub::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Int(i2)) => {
                if i2.value() == 0 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Sub(crate::halide::exprs::Sub::make(e1, e2)));
                }
            }
            (PrimeExpr::Float(f1), _) => {
                if f1.value() == 0.0 {
                    self.set_expr(PrimeExpr::Neg(crate::halide::exprs::Neg::make(e2)));
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Sub(crate::halide::exprs::Sub::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Float(f2)) => {
                if f2.value() == 0.0 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Sub(crate::halide::exprs::Sub::make(e1, e2)));
                }
            }
            _ => {
                self.set_expr(PrimeExpr::Sub(crate::halide::exprs::Sub::make(e1, e2)));
            }
        }
    }

    fn visit_div(&mut self, div: &crate::halide::exprs::Div) {
        let e1 = self.mutate_expr(div.e1());
        let e2 = self.mutate_expr(div.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(
                            i1
                                .dtype()
                                ._add(
                                    *i2.dtype()
                                ) /* we don't want float, so we use add instead of div */,
                            i1.value() / i2.value()
                        )
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(i1.dtype()._div(*f2.dtype()), (i1.value() as f64) / f2.value())
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._div(*i2.dtype()), f1.value() / (i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), _) => {
                if i1.value() == 0 {
                    self.set_expr(PrimeExpr::Int(i1.clone()));
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Int(i2)) => {
                if i2.value() == 1 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
                }
            }
            (PrimeExpr::Float(f1), _) => {
                if f1.value() == 0.0 {
                    self.set_expr(PrimeExpr::Float(f1.clone()));
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
                }
            }
            (_, PrimeExpr::Float(f2)) => {
                if f2.value() == 1.0 {
                    self.set_expr(e1);
                    self.folded = true;
                } else {
                    self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
                }
            }
            _ => {
                self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
            }
        }
    }

    fn visit_floor_div(&mut self, floor_div: &crate::halide::exprs::FloorDiv) {
        let e1 = self.mutate_expr(floor_div.e1());
        let e2 = self.mutate_expr(floor_div.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._add(*i2.dtype()), {
                            let x = i1.value();
                            let y = i2.value();
                            if x % y != 0 && (x < 0) != (y < 0) {
                                x / y - 1
                            } else {
                                x / y
                            }
                        })
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(
                            i1.dtype()._div(*f2.dtype()),
                            ((i1.value() as f64) / f2.value()).floor()
                        )
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(
                            f1.dtype()._div(*i2.dtype()),
                            (f1.value() / (i2.value() as f64)).floor()
                        )
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::Div(crate::halide::exprs::Div::make(e1, e2)));
            }
        }
    }

    fn visit_mod(&mut self, mod_: &crate::halide::exprs::Rem) {
        let e1 = self.mutate_expr(mod_.e1());
        let e2 = self.mutate_expr(mod_.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(Int::make(i1.dtype()._add(*i2.dtype()), i1.value() % i2.value()))
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(i1.dtype()._div(*f2.dtype()), (i1.value() as f64) % f2.value())
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._div(*i2.dtype()), f1.value() % (i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::Rem(crate::halide::exprs::Rem::make(e1, e2)));
            }
        }
    }

    fn visit_and(&mut self, and: &crate::halide::exprs::BitAnd) {
        let e1 = self.mutate_expr(and.e1());
        let e2 = self.mutate_expr(and.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._bitand(*i2.dtype()), i1.value() & i2.value())
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::BitAnd(crate::halide::exprs::BitAnd::make(e1, e2)));
            }
        }
    }

    fn visit_or(&mut self, or: &crate::halide::exprs::BitOr) {
        let e1 = self.mutate_expr(or.e1());
        let e2 = self.mutate_expr(or.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._bitor(*i2.dtype()), i1.value() | i2.value())
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::BitOr(crate::halide::exprs::BitOr::make(e1, e2)));
            }
        }
    }
    fn visit_xor(&mut self, xor: &crate::halide::exprs::BitXor) {
        let e1 = self.mutate_expr(xor.e1());
        let e2 = self.mutate_expr(xor.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._bitxor(*i2.dtype()), i1.value() ^ i2.value())
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::BitXor(crate::halide::exprs::BitXor::make(e1, e2)));
            }
        }
    }
    fn visit_max(&mut self, max: &crate::halide::exprs::Max) {
        let e1 = self.mutate_expr(max.e1());
        let e2 = self.mutate_expr(max.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._add(*i2.dtype()), i1.value().max(i2.value()))
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(
                            i1.dtype()._add(*f2.dtype()),
                            (i1.value() as f64).max(f2.value())
                        )
                    )
                );
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._add(*i2.dtype()), f1.value().max(i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::Max(crate::halide::exprs::Max::make(e1, e2)));
            }
        }
    }

    fn visit_min(&mut self, min: &crate::halide::exprs::Min) {
        let e1 = self.mutate_expr(min.e1());
        let e2 = self.mutate_expr(min.e2());
        match (&e1, &e2) {
            (PrimeExpr::Int(i1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Int(
                        Int::make(i1.dtype()._add(*i2.dtype()), i1.value().min(i2.value()))
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Int(i1), PrimeExpr::Float(f2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(
                            i1.dtype()._add(*f2.dtype()),
                            (i1.value() as f64).min(f2.value())
                        )
                    )
                );
                self.folded = true;
            }
            (PrimeExpr::Float(f1), PrimeExpr::Int(i2)) => {
                self.set_expr(
                    PrimeExpr::Float(
                        Float::make(f1.dtype()._add(*i2.dtype()), f1.value().min(i2.value() as f64))
                    )
                );
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::Min(crate::halide::exprs::Min::make(e1, e2)));
            }
        }
    }

    fn visit_not(&mut self, not: &crate::halide::exprs::Not) {
        let e = self.mutate_expr(not.e());
        match &e {
            PrimeExpr::Int(i) => {
                self.set_expr(PrimeExpr::Int(Int::make(i.dtype()._not(), !i.value())));
                self.folded = true;
            }
            _ => {
                self.set_expr(PrimeExpr::Not(crate::halide::exprs::Not::make(e)));
            }
        }
    }
}
