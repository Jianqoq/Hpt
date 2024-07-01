use std::{ fmt::Display, sync::Arc };

use tensor_types::dtype::Dtype;

use crate::hlir::{expr::Expr, exprs::Value};

use super::{ prime_expr::PrimeExpr, traits::{ Accepter, IRVisitor }, variable::Variable };
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Int {
    value: i64,
    dtype: Dtype,
}

impl Int {
    pub fn make(dtype: Dtype, mut value: i64) -> Self {
        value = value << (64 - dtype.bits());
        value = value >> (64 - dtype.bits());
        Int { value, dtype }
    }

    pub fn value(&self) -> i64 {
        self.value
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}

impl Accepter for Int {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_int(self);
    }
}

impl Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Into<PrimeExpr> for Int {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(self)
    }
}

impl Into<PrimeExpr> for &Int {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Int(self.clone())
    }
}

impl Into<Expr> for Int {
    fn into(self) -> Expr {
        Expr::Value(Value::make(Dtype::I64, self.value))
    }
}

impl Into<Expr> for &Int {
    fn into(self) -> Expr {
        Expr::Value(Value::make(Dtype::I64, self.value))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct UInt {
    value: u64,
    dtype: Dtype,
}

impl UInt {
    pub fn make(dtype: Dtype, mut value: u64) -> Self {
        value = value << (64 - dtype.bits());
        value = value >> (64 - dtype.bits());
        UInt { value, dtype }
    }

    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}

impl Accepter for UInt {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_uint(self);
    }
}

impl Display for UInt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Into<PrimeExpr> for UInt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(self)
    }
}

impl Into<PrimeExpr> for &UInt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::UInt(self.clone())
    }
}

macro_rules! impl_binop {
    ($lhs:ident, $rhs:ident, $res:ident, $std_op:ident, $std_op_name:ident, $op:tt) => {
        impl std::ops::$std_op for $lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: $rhs) -> Self::Output {
                $res::make(self.dtype().clone(), self.value $op rhs.value)
            }
        }
        impl std::ops::$std_op for &$lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: &$rhs) -> Self::Output {
                $res::make(self.dtype().clone(), self.value $op rhs.value)
            }
        }

        impl std::ops::$std_op<$rhs> for &$lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: $rhs) -> Self::Output {
                $res::make(self.dtype().clone(), self.value $op rhs.value)
            }
        }

        impl std::ops::$std_op<&$rhs> for $lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: &$rhs) -> Self::Output {
                $res::make(self.dtype().clone(), self.value $op rhs.value)
            }
        }
    };
}

impl_binop!(Int, Int, Int, Add, add, +);
impl_binop!(Int, Int, Int, Sub, sub, -);
impl_binop!(Int, Int, Int, Mul, mul, *);
impl_binop!(Int, Int, Int, Div, div, /);
impl_binop!(Int, Int, Int, Rem, rem, %);

impl_binop!(UInt, UInt, UInt, Add, add, +);
impl_binop!(UInt, UInt, UInt, Sub, sub, -);
impl_binop!(UInt, UInt, UInt, Mul, mul, *);
impl_binop!(UInt, UInt, UInt, Div, div, /);
impl_binop!(UInt, UInt, UInt, Rem, rem, %);

#[derive(Clone, PartialEq, Debug)]
pub struct Float {
    value: f64,
}

impl std::cmp::Eq for Float {}

impl std::hash::Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        f64::to_be_bytes(self.value).hash(state);
    }
}

impl Float {
    pub fn new(value: f64) -> Self {
        Float { value }
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn make(value: f64) -> Self {
        Float { value }
    }
}

impl Accepter for Float {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_float(self);
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Into<PrimeExpr> for Float {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(self)
    }
}

impl Into<PrimeExpr> for &Float {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Float(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Str {
    value: Arc<String>,
}

impl Str {
    pub fn new(value: String) -> Self {
        Str {
            value: value.into(),
        }
    }

    pub fn make(value: &str) -> Self {
        Str {
            value: value.to_string().into(),
        }
    }

    pub fn value(&self) -> &str {
        &self.value
    }
}

impl Accepter for Str {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_str(self);
    }
}

impl Display for Str {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Into<PrimeExpr> for Str {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Str(self)
    }
}

impl Into<PrimeExpr> for &Str {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Str(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Cast {
    expr: Arc<PrimeExpr>,
    dtype: Dtype,
}

impl Cast {
    pub fn new(expr: Arc<PrimeExpr>, dtype: Dtype) -> Self {
        Cast { expr, dtype }
    }

    pub fn make<T: Into<PrimeExpr>>(expr: T, dtype: Dtype) -> Self {
        Cast {
            expr: expr.into().into(),
            dtype,
        }
    }

    pub fn expr(&self) -> &PrimeExpr {
        &self.expr
    }

    pub fn expr_(&self) -> &Arc<PrimeExpr> {
        &self.expr
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }
}

impl Accepter for Cast {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_cast(self);
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} as {})", self.expr, self.dtype)
    }
}

impl Into<PrimeExpr> for Cast {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Cast(self)
    }
}

impl Into<PrimeExpr> for &Cast {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Cast(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Add {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Add {
    pub fn new(e1: PrimeExpr, e2: PrimeExpr) -> Self {
        Add {
            e1: e1.into(),
            e2: e2.into(),
        }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Add {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Accepter for Add {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_add(self);
    }
}

impl Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} + {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Add {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Add(self)
    }
}

impl Into<PrimeExpr> for &Add {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Add(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Sub {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Sub {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_sub(self);
    }
}

impl Sub {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Sub { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Sub {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Sub {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} - {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Sub {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Sub(self)
    }
}

impl Into<PrimeExpr> for &Sub {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Sub(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mul {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Mul {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_mul(self);
    }
}

impl Mul {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Mul { e1, e2 }
    }

    pub fn make<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(e1: A, e2: B) -> Self {
        Mul {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} * {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Mul {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Mul(self)
    }
}

impl Into<PrimeExpr> for &Mul {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Mul(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Div {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Div {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_div(self);
    }
}

impl Div {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Div { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Div {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Div {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} / {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Div {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Div(self)
    }
}

impl Into<PrimeExpr> for &Div {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Div(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Gt {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Gt {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_gt(self);
    }
}

impl Gt {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Gt { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Gt {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Gt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} > {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Gt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Gt(self)
    }
}

impl Into<PrimeExpr> for &Gt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Gt(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Ge {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Ge {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_ge(self);
    }
}

impl Ge {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Ge { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Ge {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Ge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} >= {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Ge {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Ge(self)
    }
}

impl Into<PrimeExpr> for &Ge {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Ge(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct And {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for And {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_and(self);
    }
}

impl And {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        And { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        And {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for And {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} && {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for And {
    fn into(self) -> PrimeExpr {
        PrimeExpr::And(self)
    }
}

impl Into<PrimeExpr> for &And {
    fn into(self) -> PrimeExpr {
        PrimeExpr::And(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Or {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Or {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_or(self);
    }
}

impl Or {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Or { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Or {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Or {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} || {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Or {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Or(self)
    }
}

impl Into<PrimeExpr> for &Or {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Or(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Xor {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Xor {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_xor(self);
    }
}

impl Xor {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Xor { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Xor {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Xor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ^ {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Xor {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Xor(self)
    }
}

impl Into<PrimeExpr> for &Xor {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Xor(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Not {
    e: Arc<PrimeExpr>,
}

impl Accepter for Not {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_not(self);
    }
}

impl Not {
    pub fn new(e: Arc<PrimeExpr>) -> Self {
        Not { e }
    }

    pub fn make<T: Into<PrimeExpr>>(e: T) -> Self {
        Not { e: e.into().into() }
    }

    pub fn e(&self) -> &PrimeExpr {
        &self.e
    }

    pub fn e_(&self) -> &Arc<PrimeExpr> {
        &self.e
    }
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(!{})", self.e)
    }
}

impl Into<PrimeExpr> for Not {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Not(self)
    }
}

impl Into<PrimeExpr> for &Not {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Not(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Call {
    name: Variable,
    args: Vec<Arc<PrimeExpr>>,
}

impl Accepter for Call {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_call(self);
    }
}

impl Call {
    pub fn new(name: &str, args: Vec<Arc<PrimeExpr>>) -> Self {
        Call {
            name: Variable::make(name),
            args,
        }
    }

    pub fn make(name: &str, args: &[PrimeExpr]) -> Self {
        Call {
            name: Variable::make(name),
            args: args
                .iter()
                .map(|e| (*e).clone().into())
                .collect(),
        }
    }

    pub fn name(&self) -> &Variable {
        &self.name
    }

    pub fn args(&self) -> &Vec<Arc<PrimeExpr>> {
        &self.args
    }

    pub fn args_(&self) -> &Vec<Arc<PrimeExpr>> {
        &self.args
    }
}

impl Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.name,
            self.args
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl Into<PrimeExpr> for Call {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Call(self)
    }
}

impl Into<PrimeExpr> for &Call {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Call(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Select {
    cond: Arc<PrimeExpr>,
    true_expr: Arc<PrimeExpr>,
    false_expr: Arc<PrimeExpr>,
}

impl Accepter for Select {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_select(self);
    }
}

impl Select {
    pub fn new(
        cond: Arc<PrimeExpr>,
        true_expr: Arc<PrimeExpr>,
        false_expr: Arc<PrimeExpr>
    ) -> Self {
        Select {
            cond,
            true_expr,
            false_expr,
        }
    }

    pub fn make<T: Into<PrimeExpr>>(cond: T, true_expr: T, false_expr: T) -> Self {
        Select {
            cond: cond.into().into(),
            true_expr: true_expr.into().into(),
            false_expr: false_expr.into().into(),
        }
    }

    pub fn cond(&self) -> &PrimeExpr {
        &self.cond
    }

    pub fn true_expr(&self) -> &PrimeExpr {
        &self.true_expr
    }

    pub fn false_expr(&self) -> &PrimeExpr {
        &self.false_expr
    }

    pub fn cond_(&self) -> &Arc<PrimeExpr> {
        &self.cond
    }

    pub fn true_expr_(&self) -> &Arc<PrimeExpr> {
        &self.true_expr
    }

    pub fn false_expr_(&self) -> &Arc<PrimeExpr> {
        &self.false_expr
    }
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ? {} : {})", self.cond, self.true_expr, self.false_expr)
    }
}

impl Into<PrimeExpr> for Select {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Select(self)
    }
}

impl Into<PrimeExpr> for &Select {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Select(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Load {
    name: Arc<PrimeExpr>,
    indices: Arc<PrimeExpr>,
}

impl Accepter for Load {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_load(self);
    }
}

impl Load {
    pub fn make_from_strides(name: &Variable, indices: &[Variable], strides: &[i64]) -> Self {
        if indices.len() != strides.len() {
            panic!(
                "Indices and strides must have the same length, got {:?} and {:?}",
                indices,
                strides
            );
        }
        let indices = indices
            .iter()
            .zip(strides.iter())
            .map(|(v, s)| v.clone() * Int::make(Dtype::I64, *s))
            .reduce(|a, b| a + b).unwrap();
        Load {
            name: Arc::new(name.clone().into()),
            indices: indices.into(),
        }
    }

    pub fn make<A: Into<PrimeExpr>, B: Into<PrimeExpr>>(
        name: A,
        indices: B
    ) -> Self {
        Load {
            name: Arc::new(name.into().into()),
            indices: indices.into().into(),
        }
    }

    pub fn name(&self) -> &PrimeExpr {
        &self.name
    }

    pub fn indices(&self) -> &PrimeExpr {
        &self.indices
    }

    pub fn name_(&self) -> &Arc<PrimeExpr> {
        &self.name
    }

    pub fn indices_(&self) -> &Arc<PrimeExpr> {
        &self.indices
    }
}

impl Display for Load {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]",
            self.name,
            self.indices
        )
    }
}

impl Into<PrimeExpr> for Load {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Load(self)
    }
}

impl Into<PrimeExpr> for &Load {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Load(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Let {
    name: Variable,
    e1: Arc<PrimeExpr>,
}

impl Accepter for Let {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_let(self);
    }
}

impl Let {
    pub fn new(name: Variable, e1: Arc<PrimeExpr>) -> Self {
        Let { name, e1 }
    }

    pub fn make<T: Into<PrimeExpr>>(name: &Variable, e1: T) -> Self {
        Let {
            name: name.clone(),
            e1: e1.into().into(),
        }
    }

    pub fn name(&self) -> &Variable {
        &self.name
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = {};", self.name, self.e1)
    }
}

impl Into<PrimeExpr> for Let {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Let(self)
    }
}

impl Into<PrimeExpr> for &Let {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Let(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Eq {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Eq {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Eq { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Eq {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Eq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} == {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Eq {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Eq(self)
    }
}

impl Into<PrimeExpr> for &Eq {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Eq(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Ne {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Ne {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Ne { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Ne {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Ne {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} != {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Ne {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Ne(self)
    }
}

impl Into<PrimeExpr> for &Ne {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Ne(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Lt {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Lt {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Lt { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Lt {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Lt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} < {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Lt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Lt(self)
    }
}

impl Into<PrimeExpr> for &Lt {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Lt(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Le {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Le {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_le(self);
    }
}

impl Le {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Le { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Le {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Le {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} <= {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Le {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Le(self)
    }
}

impl Into<PrimeExpr> for &Le {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Le(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mod {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Mod {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_mod(self);
    }
}

impl Mod {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Mod { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Mod {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} % {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Mod {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Mod(self)
    }
}

impl Into<PrimeExpr> for &Mod {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Mod(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Min {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Min {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_min(self);
    }
}

impl Min {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Min { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Min {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Min {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "min({}, {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Min {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Min(self)
    }
}

impl Into<PrimeExpr> for &Min {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Min(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Max {
    e1: Arc<PrimeExpr>,
    e2: Arc<PrimeExpr>,
}

impl Accepter for Max {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_max(self);
    }
}

impl Max {
    pub fn new(e1: Arc<PrimeExpr>, e2: Arc<PrimeExpr>) -> Self {
        Max { e1, e2 }
    }

    pub fn make<T: Into<PrimeExpr>>(e1: T, e2: T) -> Self {
        Max {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &PrimeExpr {
        &self.e1
    }

    pub fn e2(&self) -> &PrimeExpr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<PrimeExpr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<PrimeExpr> {
        &self.e2
    }
}

impl Display for Max {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "max({}, {})", self.e1, self.e2)
    }
}

impl Into<PrimeExpr> for Max {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Max(self)
    }
}

impl Into<PrimeExpr> for &Max {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Max(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Reduce {
    identity: Arc<PrimeExpr>,
    start: Arc<Vec<PrimeExpr>>,
    end: Arc<Vec<PrimeExpr>>,
    step: Arc<Vec<PrimeExpr>>,
    loop_var: Arc<Vec<PrimeExpr>>,
    expr: Arc<PrimeExpr>,
}

impl Accepter for Reduce {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_reduce(self);
    }
}

impl Reduce {
    pub fn new(
        expr: Arc<PrimeExpr>,
        identity: Arc<PrimeExpr>,
        start: Arc<Vec<PrimeExpr>>,
        end: Arc<Vec<PrimeExpr>>,
        step: Arc<Vec<PrimeExpr>>,
        loop_var: Arc<Vec<PrimeExpr>>
    ) -> Self {
        Reduce {
            expr,
            identity,
            start,
            end,
            step,
            loop_var
        }
    }

    pub fn make<T: Into<PrimeExpr>>(
        expr: T,
        identity: T,
        start: Vec<T>,
        end: Vec<T>,
        step: Vec<T>,
        loop_var: Vec<T>
    ) -> Self {
        Reduce {
            expr: expr.into().into(),
            identity: identity.into().into(),
            start: Arc::new(start.into_iter().map(|e| e.into().into()).collect()),
            end: Arc::new(end.into_iter().map(|e| e.into().into()).collect()),
            step: Arc::new(step.into_iter().map(|e| e.into().into()).collect()),
            loop_var: Arc::new(loop_var.into_iter().map(|e| e.into().into()).collect())
        }
    }

    pub fn identity(&self) -> &PrimeExpr {
        &self.identity
    }

    pub fn start(&self) -> &Vec<PrimeExpr> {
        &self.start
    }

    pub fn end(&self) -> &Vec<PrimeExpr> {
        &self.end
    }

    pub fn step(&self) -> &Vec<PrimeExpr> {
        &self.step
    }

    pub fn loop_var(&self) -> &Vec<PrimeExpr> {
        &self.loop_var
    }

    pub fn identity_(&self) -> &Arc<PrimeExpr> {
        &self.identity
    }

    pub fn start_(&self) -> &Arc<Vec<PrimeExpr>> {
        &self.start
    }

    pub fn end_(&self) -> &Arc<Vec<PrimeExpr>> {
        &self.end
    }

    pub fn step_(&self) -> &Arc<Vec<PrimeExpr>> {
        &self.step
    }

    pub fn loop_var_(&self) -> &Arc<Vec<PrimeExpr>> {
        &self.loop_var
    }
    pub fn expr(&self) -> &PrimeExpr {
        &self.expr
    }
    pub fn expr_(&self) -> &Arc<PrimeExpr> {
        &self.expr
    }
}

impl Display for Reduce {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "reduce",
        )
    }
}

impl Into<PrimeExpr> for Reduce {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Reduce(self)
    }
}

impl Into<PrimeExpr> for &Reduce {
    fn into(self) -> PrimeExpr {
        PrimeExpr::Reduce(self.clone())
    }
}