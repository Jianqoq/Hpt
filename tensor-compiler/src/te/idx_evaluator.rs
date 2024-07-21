use std::collections::HashMap;

use crate::halide::{ prime_expr::PrimeExpr, traits::IRMutVisitor };

pub struct IdxEvaluator<'a> {
    value: i64,
    map: &'a HashMap<String, i64>,
}

impl<'a> IdxEvaluator<'a> {
    pub fn new(map: &'a HashMap<String, i64>) -> Self {
        Self {
            value: 0,
            map,
        }
    }
    pub fn eval(&mut self, expr: &PrimeExpr) -> i64 {
        self.visit_expr(expr);
        self.value
    }
}

impl<'a> IRMutVisitor for IdxEvaluator<'a> {
    fn visit_expr(&mut self, expr: &PrimeExpr) {
        match expr {
            PrimeExpr::Int(int) => {
                self.value = int.value();
            }
            PrimeExpr::Float(float) => {
                self.value = float.value() as i64;
            }
            PrimeExpr::UInt(uint) => {
                self.value = uint.value() as i64;
            }
            PrimeExpr::Variable(var) => {
                if let Some(value) = self.map.get(var.name()) {
                    self.value = *value;
                } else {
                    panic!("Variable {} not found in the map", var.name());
                }
            }
            PrimeExpr::Add(add) => {
                self.visit_expr(add.e1());
                let e1 = self.value;
                self.visit_expr(add.e2());
                let e2 = self.value;
                self.value = e1 + e2;
            }
            PrimeExpr::Sub(sub) => {
                self.visit_expr(sub.e1());
                let e1 = self.value;
                self.visit_expr(sub.e2());
                let e2 = self.value;
                self.value = e1 - e2;
            }
            PrimeExpr::Mul(mul) => {
                self.visit_expr(mul.e1());
                let e1 = self.value;
                self.visit_expr(mul.e2());
                let e2 = self.value;
                self.value = e1 * e2;
            }
            PrimeExpr::Div(div) => {
                self.visit_expr(div.e1());
                let e1 = self.value;
                self.visit_expr(div.e2());
                let e2 = self.value;
                self.value = e1 / e2;
            }
            PrimeExpr::Rem(rem) => {
                self.visit_expr(rem.e1());
                let e1 = self.value;
                self.visit_expr(rem.e2());
                let e2 = self.value;
                self.value = e1 % e2;
            }
            PrimeExpr::Min(min) => {
                self.visit_expr(min.e1());
                let e1 = self.value;
                self.visit_expr(min.e2());
                let e2 = self.value;
                self.value = e1.min(e2);
            }
            PrimeExpr::Max(max) => {
                self.visit_expr(max.e1());
                let e1 = self.value;
                self.visit_expr(max.e2());
                let e2 = self.value;
                self.value = e1.max(e2);
            }
            PrimeExpr::FloorDiv(floor_div) => {
                self.visit_expr(floor_div.e1());
                let a = self.value;
                self.visit_expr(floor_div.e2());
                let b = self.value;
                if (a < 0) != (b < 0) && a % b != 0 {
                    self.value = a / b - 1;
                } else {
                    self.value = a / b;
                }
            }
            PrimeExpr::Shl(shl) => {
                self.visit_expr(shl.e1());
                let e1 = self.value;
                self.visit_expr(shl.e2());
                let e2 = self.value;
                self.value = e1 << e2;
            }
            PrimeExpr::Shr(shr) => {
                self.visit_expr(shr.e1());
                let e1 = self.value;
                self.visit_expr(shr.e2());
                let e2 = self.value;
                self.value = e1 >> e2;
            }
            _ => unreachable!(),
        }
    }
}
