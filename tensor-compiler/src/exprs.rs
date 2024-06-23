use std::{fmt::Display, sync::Arc};

use tensor_types::dtype::Dtype;

use super::{
    expr::Expr,
    r#type::{HalideirTypeCode, Type},
    traits::{Accepter, IRVisitor},
    variable::Variable,
};
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Int {
    value: i64,
    r#type: Type,
}

impl Int {
    pub fn make(r#type: Type, mut value: i64) -> Self {
        value = value << (64 - r#type.bits());
        value = value >> (64 - r#type.bits());
        Int { value, r#type }
    }

    pub fn value(&self) -> i64 {
        self.value
    }

    pub fn r#type(&self) -> &Type {
        &self.r#type
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

impl Into<Expr> for Int {
    fn into(self) -> Expr {
        Expr::Int(self)
    }
}

impl Into<Expr> for &Int {
    fn into(self) -> Expr {
        Expr::Int(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct UInt {
    value: u64,
    r#type: Type,
}

impl UInt {
    pub fn make(r#type: Type, mut value: u64) -> Self {
        value = value << (64 - r#type.bits());
        value = value >> (64 - r#type.bits());
        UInt { value, r#type }
    }

    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn r#type(&self) -> &Type {
        &self.r#type
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

impl Into<Expr> for UInt {
    fn into(self) -> Expr {
        Expr::UInt(self)
    }
}

impl Into<Expr> for &UInt {
    fn into(self) -> Expr {
        Expr::UInt(self.clone())
    }
}

macro_rules! impl_binop {
    ($lhs: ident, $rhs: ident, $res: ident, $std_op: ident, $std_op_name: ident, $op: tt) => {
        impl std::ops::$std_op for $lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: $rhs) -> Self::Output {
                $res::make(self.r#type().clone(), self.value $op rhs.value)
            }
        }
        impl std::ops::$std_op for &$lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: &$rhs) -> Self::Output {
                $res::make(self.r#type().clone(), self.value $op rhs.value)
            }
        }

        impl std::ops::$std_op<$rhs> for &$lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: $rhs) -> Self::Output {
                $res::make(self.r#type().clone(), self.value $op rhs.value)
            }
        }

        impl std::ops::$std_op<&$rhs> for $lhs {
            type Output = $res;

            fn $std_op_name(self, rhs: &$rhs) -> Self::Output {
                $res::make(self.r#type().clone(), self.value $op rhs.value)
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

impl Into<Expr> for Float {
    fn into(self) -> Expr {
        Expr::Float(self)
    }
}

impl Into<Expr> for &Float {
    fn into(self) -> Expr {
        Expr::Float(self.clone())
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

impl Into<Expr> for Str {
    fn into(self) -> Expr {
        Expr::Str(self)
    }
}

impl Into<Expr> for &Str {
    fn into(self) -> Expr {
        Expr::Str(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Cast {
    expr: Arc<Expr>,
    dtype: Dtype,
}

impl Cast {
    pub fn new(expr: Arc<Expr>, dtype: Dtype) -> Self {
        Cast { expr, dtype }
    }

    pub fn make<T: Into<Expr>>(expr: T, dtype: Dtype) -> Self {
        Cast {
            expr: expr.into().into(),
            dtype,
        }
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn expr_(&self) -> &Arc<Expr> {
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

impl Into<Expr> for Cast {
    fn into(self) -> Expr {
        Expr::Cast(self)
    }
}

impl Into<Expr> for &Cast {
    fn into(self) -> Expr {
        Expr::Cast(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Add {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Add {
    pub fn new(e1: Expr, e2: Expr) -> Self {
        Add {
            e1: e1.into(),
            e2: e2.into(),
        }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Add {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
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

impl Into<Expr> for Add {
    fn into(self) -> Expr {
        Expr::Add(self)
    }
}

impl Into<Expr> for &Add {
    fn into(self) -> Expr {
        Expr::Add(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Sub {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Sub {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_sub(self);
    }
}

impl Sub {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Sub { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Sub {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Sub {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} - {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Sub {
    fn into(self) -> Expr {
        Expr::Sub(self)
    }
}

impl Into<Expr> for &Sub {
    fn into(self) -> Expr {
        Expr::Sub(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mul {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Mul {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_mul(self);
    }
}

impl Mul {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Mul { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Mul {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} * {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Mul {
    fn into(self) -> Expr {
        Expr::Mul(self)
    }
}

impl Into<Expr> for &Mul {
    fn into(self) -> Expr {
        Expr::Mul(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Div {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Div {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_div(self);
    }
}

impl Div {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Div { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Div {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Div {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} / {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Div {
    fn into(self) -> Expr {
        Expr::Div(self)
    }
}

impl Into<Expr> for &Div {
    fn into(self) -> Expr {
        Expr::Div(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Gt {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Gt {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_gt(self);
    }
}

impl Gt {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Gt { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Gt {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Gt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} > {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Gt {
    fn into(self) -> Expr {
        Expr::Gt(self)
    }
}

impl Into<Expr> for &Gt {
    fn into(self) -> Expr {
        Expr::Gt(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Ge {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Ge {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_ge(self);
    }
}

impl Ge {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Ge { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Ge {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Ge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} >= {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Ge {
    fn into(self) -> Expr {
        Expr::Ge(self)
    }
}

impl Into<Expr> for &Ge {
    fn into(self) -> Expr {
        Expr::Ge(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct And {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for And {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_and(self);
    }
}

impl And {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        And { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        And {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for And {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} && {})", self.e1, self.e2)
    }
}

impl Into<Expr> for And {
    fn into(self) -> Expr {
        Expr::And(self)
    }
}

impl Into<Expr> for &And {
    fn into(self) -> Expr {
        Expr::And(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Or {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Or {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_or(self);
    }
}

impl Or {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Or { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Or {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Or {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} || {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Or {
    fn into(self) -> Expr {
        Expr::Or(self)
    }
}

impl Into<Expr> for &Or {
    fn into(self) -> Expr {
        Expr::Or(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Xor {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Xor {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_xor(self);
    }
}

impl Xor {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Xor { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Xor {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Xor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ^ {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Xor {
    fn into(self) -> Expr {
        Expr::Xor(self)
    }
}

impl Into<Expr> for &Xor {
    fn into(self) -> Expr {
        Expr::Xor(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Not {
    e: Arc<Expr>,
}

impl Accepter for Not {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_not(self);
    }
}

impl Not {
    pub fn new(e: Arc<Expr>) -> Self {
        Not { e }
    }

    pub fn make<T: Into<Expr>>(e: T) -> Self {
        Not { e: e.into().into() }
    }

    pub fn e(&self) -> &Expr {
        &self.e
    }

    pub fn e_(&self) -> &Arc<Expr> {
        &self.e
    }
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(!{})", self.e)
    }
}

impl Into<Expr> for Not {
    fn into(self) -> Expr {
        Expr::Not(self)
    }
}

impl Into<Expr> for &Not {
    fn into(self) -> Expr {
        Expr::Not(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Call {
    name: Variable,
    args: Vec<Arc<Expr>>,
}

impl Accepter for Call {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_call(self);
    }
}

impl Call {
    pub fn new(name: &str, args: Vec<Arc<Expr>>) -> Self {
        Call {
            name: Variable::make(name),
            args,
        }
    }

    pub fn make(name: &str, args: &[Expr]) -> Self {
        Call {
            name: Variable::make(name),
            args: args.iter().map(|e| (*e).clone().into()).collect(),
        }
    }

    pub fn name(&self) -> &Variable {
        &self.name
    }

    pub fn args(&self) -> &Vec<Arc<Expr>> {
        &self.args
    }

    pub fn args_(&self) -> &Vec<Arc<Expr>> {
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

impl Into<Expr> for Call {
    fn into(self) -> Expr {
        Expr::Call(self)
    }
}

impl Into<Expr> for &Call {
    fn into(self) -> Expr {
        Expr::Call(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Select {
    cond: Arc<Expr>,
    true_expr: Arc<Expr>,
    false_expr: Arc<Expr>,
}

impl Accepter for Select {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_select(self);
    }
}

impl Select {
    pub fn new(cond: Arc<Expr>, true_expr: Arc<Expr>, false_expr: Arc<Expr>) -> Self {
        Select {
            cond,
            true_expr,
            false_expr,
        }
    }

    pub fn make<T: Into<Expr>>(cond: T, true_expr: T, false_expr: T) -> Self {
        Select {
            cond: cond.into().into(),
            true_expr: true_expr.into().into(),
            false_expr: false_expr.into().into(),
        }
    }

    pub fn cond(&self) -> &Expr {
        &self.cond
    }

    pub fn true_expr(&self) -> &Expr {
        &self.true_expr
    }

    pub fn false_expr(&self) -> &Expr {
        &self.false_expr
    }

    pub fn cond_(&self) -> &Arc<Expr> {
        &self.cond
    }

    pub fn true_expr_(&self) -> &Arc<Expr> {
        &self.true_expr
    }

    pub fn false_expr_(&self) -> &Arc<Expr> {
        &self.false_expr
    }
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "({} ? {} : {})",
            self.cond, self.true_expr, self.false_expr
        )
    }
}

impl Into<Expr> for Select {
    fn into(self) -> Expr {
        Expr::Select(self)
    }
}

impl Into<Expr> for &Select {
    fn into(self) -> Expr {
        Expr::Select(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Load {
    name: Arc<Expr>,
    indices: Arc<Expr>,
}

impl Accepter for Load {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_load(self);
    }
}

impl Load {
    pub fn new(name: Arc<Expr>, indices: Arc<Expr>) -> Self {
        Load { name, indices }
    }

    pub fn make_from_strides(name: &Variable, indices: &[Variable], strides: &[i64]) -> Self {
        if indices.len() != strides.len() {
            panic!("Indices and strides must have the same length, got {:?} and {:?}", indices, strides);
        }
        let sum = strides
            .iter()
            .zip(indices.iter())
            .map(|(stride, index)| {
                index * Int::make(Type::new(HalideirTypeCode::Int, 64, 1), *stride)
            })
            .reduce(|acc, e| acc + e)
            .expect("Failed to reduce");
        Load {
            name: Arc::new(name.clone().into()),
            indices: sum.into(),
        }
    }

    pub fn make<T: Into<Expr>>(name: T, indices: T) -> Self {
        Load {
            name: Arc::new(name.into().into()),
            indices: indices.into().into(),
        }
    }

    pub fn name(&self) -> &Expr {
        &self.name
    }

    pub fn indices(&self) -> &Expr {
        &self.indices
    }

    pub fn name_(&self) -> &Arc<Expr> {
        &self.name
    }

    pub fn indices_(&self) -> &Arc<Expr> {
        &self.indices
    }
}

impl Display for Load {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[{}]", self.name, self.indices)
    }
}

impl Into<Expr> for Load {
    fn into(self) -> Expr {
        Expr::Load(self)
    }
}

impl Into<Expr> for &Load {
    fn into(self) -> Expr {
        Expr::Load(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Let {
    name: Variable,
    e1: Arc<Expr>,
}

impl Accepter for Let {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_let(self);
    }
}

impl Let {
    pub fn new(name: Variable, e1: Arc<Expr>) -> Self {
        Let { name, e1 }
    }

    pub fn make<T: Into<Expr>>(name: &Variable, e1: T) -> Self {
        Let {
            name: name.clone(),
            e1: e1.into().into(),
        }
    }

    pub fn name(&self) -> &Variable {
        &self.name
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = {};", self.name, self.e1)
    }
}

impl Into<Expr> for Let {
    fn into(self) -> Expr {
        Expr::Let(self)
    }
}

impl Into<Expr> for &Let {
    fn into(self) -> Expr {
        Expr::Let(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Eq {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Eq {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Eq { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Eq {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Eq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} == {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Eq {
    fn into(self) -> Expr {
        Expr::Eq(self)
    }
}

impl Into<Expr> for &Eq {
    fn into(self) -> Expr {
        Expr::Eq(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Ne {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Ne {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Ne { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Ne {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Ne {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} != {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Ne {
    fn into(self) -> Expr {
        Expr::Ne(self)
    }
}

impl Into<Expr> for &Ne {
    fn into(self) -> Expr {
        Expr::Ne(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Lt {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Lt {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Lt { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Lt {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Lt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} < {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Lt {
    fn into(self) -> Expr {
        Expr::Lt(self)
    }
}

impl Into<Expr> for &Lt {
    fn into(self) -> Expr {
        Expr::Lt(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Le {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Le {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_le(self);
    }
}

impl Le {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Le { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Le {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Le {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} <= {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Le {
    fn into(self) -> Expr {
        Expr::Le(self)
    }
}

impl Into<Expr> for &Le {
    fn into(self) -> Expr {
        Expr::Le(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Mod {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Mod {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_mod(self);
    }
}

impl Mod {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Mod { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Mod {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} % {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Mod {
    fn into(self) -> Expr {
        Expr::Mod(self)
    }
}

impl Into<Expr> for &Mod {
    fn into(self) -> Expr {
        Expr::Mod(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Min {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Min {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_min(self);
    }
}

impl Min {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Min { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Min {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Min {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "min({}, {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Min {
    fn into(self) -> Expr {
        Expr::Min(self)
    }
}

impl Into<Expr> for &Min {
    fn into(self) -> Expr {
        Expr::Min(self.clone())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Max {
    e1: Arc<Expr>,
    e2: Arc<Expr>,
}

impl Accepter for Max {
    fn accept<V: IRVisitor>(&self, visitor: &V) {
        visitor.visit_max(self);
    }
}

impl Max {
    pub fn new(e1: Arc<Expr>, e2: Arc<Expr>) -> Self {
        Max { e1, e2 }
    }

    pub fn make<T: Into<Expr>>(e1: T, e2: T) -> Self {
        Max {
            e1: e1.into().into(),
            e2: e2.into().into(),
        }
    }

    pub fn e1(&self) -> &Expr {
        &self.e1
    }

    pub fn e2(&self) -> &Expr {
        &self.e2
    }

    pub fn e1_(&self) -> &Arc<Expr> {
        &self.e1
    }

    pub fn e2_(&self) -> &Arc<Expr> {
        &self.e2
    }
}

impl Display for Max {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "max({}, {})", self.e1, self.e2)
    }
}

impl Into<Expr> for Max {
    fn into(self) -> Expr {
        Expr::Max(self)
    }
}

impl Into<Expr> for &Max {
    fn into(self) -> Expr {
        Expr::Max(self.clone())
    }
}
