use std::{ fmt::Display, sync::Arc };

use tensor_common::{
    axis,
    layout::Layout,
    shape::Shape,
    shape_utils::{ get_broadcast_axes_from, predict_broadcast_shape, try_pad_shape },
};
use tensor_types::dtype::Dtype;

use crate::{
    halide::{ exprs::{ Int, Load }, prime_expr::PrimeExpr, variable::Variable },
    op::OpType,
    registry::Closures,
};

use super::{
    _value::_Value,
    expr::Expr,
    func_type::Type,
    traits::{
        HlirAccepterMut,
        HlirAccepterMutate,
        HlirAcceptor,
        HlirMutVisitor,
        HlirMutateVisitor,
        HlirVisitor,
        IntoVar,
    },
};

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Value {
    dtype: Dtype,
    value: _Value,
}

impl Value {
    pub fn make<T: Into<_Value>>(dtype: Dtype, value: T) -> Self {
        Self {
            dtype,
            value: value.into(),
        }
    }
}

impl HlirAcceptor for Value {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_value(self);
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Tuple {
    values: Arc<Vec<Expr>>,
}

impl Tuple {
    pub fn make<T: IntoIterator<Item: Into<Expr>>>(values: T) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|x| x.into().into())
                .collect::<Vec<Expr>>()
                .into(),
        }
    }
    pub fn values(&self) -> &[Expr] {
        &self.values
    }
}

impl HlirAcceptor for Tuple {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_tuple(self);
    }
}

impl Display for Tuple {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "({})",
            self.values
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Str {
    value: Arc<String>,
}

impl Str {
    pub fn make(value: &str) -> Self {
        Self {
            value: Arc::new(value.into()),
        }
    }
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl HlirAcceptor for Str {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_str(self);
    }
}

impl Display for Str {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl HlirAcceptor for Variable {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_variable(self);
    }
}

impl HlirAccepterMut for Variable {
    fn accept_mut<V: HlirMutVisitor>(&self, visitor: &mut V) {
        visitor.visit_variable(self);
    }
}

impl HlirAccepterMutate for Variable {
    fn accept_mutate<V: HlirMutateVisitor>(&self, visitor: &mut V) {
        visitor.visit_variable(self);
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Cast {
    expr: Arc<Expr>,
    dtype: Dtype,
}

impl Cast {
    pub fn make<T: Into<Expr>>(expr: T, dtype: Dtype) -> Self {
        Self { expr: expr.into().into(), dtype }
    }
    pub fn expr(&self) -> &Expr {
        &self.expr
    }
    pub fn expr_(&self) -> &Arc<Expr> {
        &self.expr
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

impl HlirAcceptor for Cast {
    fn accept<V: HlirVisitor>(&self, visitor: &V) {
        visitor.visit_cast(self);
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} as {})", self.expr, self.dtype)
    }
}

macro_rules! impl_binop {
    ($struct:ident, $visit_method:ident) => {
        impl $struct {
            pub fn make<T: Into<Expr>, U: Into<Expr>>(lhs: T, rhs: U) -> Self {
                Self { lhs: lhs.into().into(), rhs: rhs.into().into() }
            }
            pub fn lhs(&self) -> &Expr {
                &self.lhs
            }
            pub fn rhs(&self) -> &Expr {
                &self.rhs
            }
            pub fn lhs_(&self) -> &Arc<Expr> {
                &self.lhs
            }
            pub fn rhs_(&self) -> &Arc<Expr> {
                &self.rhs
            }
        }

        impl HlirAcceptor for $struct {
            fn accept<V: HlirVisitor>(&self, visitor: &V) {
                visitor.$visit_method(self);
            }
        }
    };
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct OpNode {
    name: Variable,
    args: Option<
        Arc<Vec<(String /* arg name */, String /* arg type */, String /* arg description */)>>
    >,
    num_inputs: i64,
    fn_type: Arc<Type>,
    registry_idx: usize,
    op_type: OpType,
}

impl OpNode {
    pub fn new<T: Into<Variable>>(name: T) -> Self {
        Self {
            name: name.into(),
            args: None,
            registry_idx: 0,
            num_inputs: 0,
            fn_type: Arc::new(Type::None),
            op_type: OpType::Opaque,
        }
    }
    pub fn var(&self) -> &Variable {
        &self.name
    }
    pub fn name(&self) -> &str {
        &self.name.name
    }
    pub fn registry_idx(&self) -> usize {
        self.registry_idx
    }
    pub fn set_registry_idx(&mut self, idx: usize) {
        self.registry_idx = idx;
    }

    pub fn set_num_inputs(&mut self, num_inputs: i64) {
        self.num_inputs = num_inputs;
    }
    pub fn add_argument<A: Into<String>, B: Into<String>, C: Into<String>>(
        &mut self,
        arg_name: A,
        arg_type: B,
        arg_desc: C
    ) {
        if let Some(args) = self.args.as_mut() {
            Arc::make_mut(args).push((arg_name.into(), arg_type.into(), arg_desc.into()));
        } else {
            self.args = Some(vec![(arg_name.into(), arg_type.into(), arg_desc.into())].into());
        }
    }

    pub fn set_op_type(&mut self, op_type: OpType) {
        self.op_type = op_type;
    }
}

impl Display for OpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "OpNode({})", self.name)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Add {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} + {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Sub {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Sub {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} - {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Mul {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} * {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Div {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Div {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} / {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Mod {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} % {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Min {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Min {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "min({}, {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Max {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Max {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "max({}, {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Eq {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Eq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} == {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Ne {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Ne {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} != {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Lt {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Lt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} < {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Le {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Le {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} <= {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Gt {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Gt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} > {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Ge {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Ge {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} >= {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct And {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for And {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} && {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Or {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Or {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} || {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Xor {
    lhs: Arc<Expr>,
    rhs: Arc<Expr>,
}

impl Display for Xor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ^ {})", self.lhs, self.rhs)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Not {
    expr: Arc<Expr>,
}

impl Not {
    pub fn make<T: Into<Expr>>(expr: T) -> Self {
        Self { expr: expr.into().into() }
    }
    pub fn expr(&self) -> &Expr {
        &self.expr
    }
    pub fn expr_(&self) -> &Arc<Expr> {
        &self.expr
    }
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "!{}", self.expr)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Call {
    op: Arc<Expr>,
    args: Arc<Vec<Expr>>,
}

impl Call {
    pub fn make<T: Into<Expr>, U: IntoIterator<Item: Into<Expr>>>(name: T, args: U) -> Self {
        Self {
            op: Arc::new(name.into()),
            args: args
                .into_iter()
                .map(|x| x.into().into())
                .collect::<Vec<Expr>>()
                .into(),
        }
    }
    pub fn name(&self) -> &Expr {
        &self.op
    }
    pub fn args(&self) -> &[Expr] {
        &self.args
    }
}

impl Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}({})",
            self.op,
            self.args
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Select {
    cond: Arc<Expr>,
    true_value: Arc<Expr>,
    false_value: Arc<Expr>,
}

impl Select {
    pub fn make<T: Into<Expr>, U: Into<Expr>, V: Into<Expr>>(
        cond: T,
        true_value: U,
        false_value: V
    ) -> Self {
        Self {
            cond: cond.into().into(),
            true_value: true_value.into().into(),
            false_value: false_value.into().into(),
        }
    }
    pub fn cond(&self) -> &Expr {
        &self.cond
    }
    pub fn true_value(&self) -> &Expr {
        &self.true_value
    }
    pub fn false_value(&self) -> &Expr {
        &self.false_value
    }
    pub fn cond_(&self) -> &Arc<Expr> {
        &self.cond
    }
    pub fn true_value_(&self) -> &Arc<Expr> {
        &self.true_value
    }
    pub fn false_value_(&self) -> &Arc<Expr> {
        &self.false_value
    }
}

impl Display for Select {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} ? {} : {})", self.cond, self.true_value, self.false_value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Let {
    var: Arc<Variable>,
    value: Arc<Expr>,
    body: Arc<Expr>,
}

impl Let {
    pub fn make<U: Into<Expr>, T: IntoVar, C: Into<Expr>>(var: T, value: U, body: C) -> Self {
        Self {
            var: var.into_var().into(),
            value: value.into().into(),
            body: body.into().into(),
        }
    }
    pub fn make_from_expr<T: Into<Variable>, U: Into<Expr>, C: Into<Expr>>(
        var: T,
        value: U,
        body: C
    ) -> Self {
        Self {
            var: var.into().into(),
            value: value.into().into(),
            body: body.into().into(),
        }
    }
    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn value(&self) -> &Expr {
        &self.value
    }
    pub fn var_(&self) -> &Arc<Variable> {
        &self.var
    }
    pub fn value_(&self) -> &Arc<Expr> {
        &self.value
    }
    pub fn body(&self) -> &Expr {
        &self.body
    }
    pub fn body_(&self) -> &Arc<Expr> {
        &self.body
    }
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.body.as_ref() {
            Expr::Let(_) | Expr::For(_) | Expr::While(_) | Expr::If(_) =>
                write!(f, "let {} = {};\n{}", self.var, self.value, self.body),
            Expr::None => write!(f, "let {} = {}", self.var, self.value),
            _ => write!(f, "let {} = {} in {}", self.var, self.value, self.body),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Tensor {
    name: Variable,
    shape: Shape,
    dtype: Dtype,
}

impl Tensor {
    pub fn make<T: IntoVar>(name: T, layout: Shape, dtype: Dtype) -> Self {
        Self {
            name: name.into_var(),
            shape: layout,
            dtype,
        }
    }

    pub fn name(&self) -> &Variable {
        &self.name
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor({}, shape={:?}, {})", self.name, self.shape.inner(), self.dtype)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct ComputeNode {
    compute_expr: PrimeExpr,
    shape: Shape,
}

impl ComputeNode {
    pub fn make_from_tensor<T: Into<Tensor>>(tensor: T) -> Self {
        todo!()
    }
    pub fn make_binop<A: Into<Tensor>, B: Into<Tensor>>(func: Closures, lhs: A, rhs: B) -> Self {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let a_shape = lhs.shape();
        let b_shape = rhs.shape();
        let broadcast_shape = predict_broadcast_shape(a_shape, b_shape).unwrap();
        let axis = (0..broadcast_shape.len())
            .map(|i| Variable::new(format!("ax{}", i)).into())
            .collect::<Vec<PrimeExpr>>();
        let one: PrimeExpr = Int::make(Dtype::I64, 1).into();
        let a_axes = get_broadcast_axes_from(&a_shape, &broadcast_shape).unwrap();
        let b_axes = get_broadcast_axes_from(&b_shape, &broadcast_shape).unwrap();

        let a_axes_var = axis
            .iter()
            .enumerate()
            .map(|(i, axis)| {
                if a_axes.contains(&i) { one.clone() } else { axis.clone() }
            })
            .collect::<Vec<PrimeExpr>>();
        let b_axes_var = axis
            .iter()
            .enumerate().rev()
            .map(|(i, axis)| {
                if b_axes.contains(&i) { one.clone() } else { axis.clone() }
            })
            .collect::<Vec<PrimeExpr>>();
        let a_load = Load::make(lhs.name(), a_axes_var);
        let b_load = Load::make(rhs.name(), b_axes_var);
        let expr = func.get_binop_expr(a_load, b_load);
        Self {
            compute_expr: expr.into(),
            shape: broadcast_shape,
        }
    }
    pub fn compute_expr(&self) -> &PrimeExpr {
        &self.compute_expr
    }
}

impl Display for ComputeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.compute_expr)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct TensorType {
    dtype: Dtype,
    layout: Arc<Layout>,
}

impl TensorType {
    pub fn make(dtype: Dtype, layout: Layout) -> Self {
        Self {
            dtype,
            layout: Arc::new(layout),
        }
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Tensor(dtype={}, shape={:?}, strides={:?})",
            self.dtype,
            self.layout.shape().inner(),
            self.layout.strides().inner()
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Alloc {
    shape: Arc<Expr>,
    dtype: Dtype,
}

impl Alloc {
    pub fn make<T: Into<Expr>>(shape: T, dtype: Dtype) -> Self {
        Self {
            shape: shape.into().into(),
            dtype,
        }
    }
    pub fn shape(&self) -> &Expr {
        &self.shape
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
}

impl Display for Alloc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "alloc({}, {})", self.shape, self.dtype)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct If {
    cond: Arc<Expr>,
    then: Arc<Expr>,
    else_: Arc<Expr>,
}

impl If {
    pub fn make<T: Into<Expr>, U: Into<Expr>, V: Into<Expr>>(cond: T, then: U, else_: V) -> Self {
        Self {
            cond: cond.into().into(),
            then: then.into().into(),
            else_: else_.into().into(),
        }
    }
    pub fn cond(&self) -> &Expr {
        &self.cond
    }
    pub fn then(&self) -> &Expr {
        &self.then
    }
    pub fn else_(&self) -> &Expr {
        &self.else_
    }
    pub fn cond_(&self) -> &Arc<Expr> {
        &self.cond
    }
    pub fn then_(&self) -> &Arc<Expr> {
        &self.then
    }
    pub fn else__(&self) -> &Arc<Expr> {
        &self.else_
    }
}

impl Display for If {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "if {} {{\n{}}} else {{\n{}}}", self.cond, self.then, self.else_)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct For {
    var: Arc<Variable>,
    start: Arc<Expr>,
    end: Arc<Expr>,
    step: Arc<Expr>,
    body: Arc<Expr>,
}

impl For {
    pub fn make<T: IntoVar, U: Into<Expr>, V: Into<Expr>, W: Into<Expr>, X: Into<Expr>>(
        var: T,
        start: U,
        end: V,
        step: W,
        body: X
    ) -> Self {
        Self {
            var: var.into_var().into(),
            start: start.into().into(),
            end: end.into().into(),
            step: step.into().into(),
            body: body.into().into(),
        }
    }
    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn start(&self) -> &Expr {
        &self.start
    }
    pub fn end(&self) -> &Expr {
        &self.end
    }
    pub fn step(&self) -> &Expr {
        &self.step
    }
    pub fn body(&self) -> &Expr {
        &self.body
    }
}

impl Display for For {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "for {} in {}..{} {{\n{}\n}}", self.var, self.start, self.end, self.body)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct While {
    cond: Arc<Expr>,
    body: Arc<Expr>,
}

impl While {
    pub fn make<T: Into<Expr>, U: Into<Expr>>(cond: T, body: U) -> Self {
        Self {
            cond: cond.into().into(),
            body: body.into().into(),
        }
    }
    pub fn cond(&self) -> &Expr {
        &self.cond
    }
    pub fn body(&self) -> &Expr {
        &self.body
    }
}

impl Display for While {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "while {} {{\n{}\n}}", self.cond, self.body)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Function {
    name: Arc<String>,
    args: Arc<Vec<Variable>>,
    return_type: Arc<Expr>,
    body: Arc<Expr>,
}

impl Function {
    pub fn make<T: Into<String>, U: IntoIterator<Item: IntoVar>, W: Into<Expr>>(
        name: T,
        args: U,
        return_type: &Type,
        body: W
    ) -> Self {
        let ret_type: Expr = return_type.into();
        Self {
            name: Arc::new(name.into()),
            args: args
                .into_iter()
                .map(|x| x.into_var())
                .collect::<Vec<Variable>>()
                .into(),
            return_type: ret_type.into(),
            body: body.into().into(),
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn args(&self) -> &[Variable] {
        &self.args
    }
    pub fn return_type(&self) -> &Expr {
        &self.return_type
    }
    pub fn body(&self) -> &Expr {
        &self.body
    }
    pub fn body_mut(&mut self) -> &mut Expr {
        Arc::make_mut(&mut self.body)
    }

    pub fn set_body<T: Into<Expr>>(&mut self, body: T) {
        self.body = body.into().into();
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "fn {}({}) -> {} {{\n{}\n}}",
            self.name,
            self.args
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join(", "),
            self.return_type,
            self.body
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Slice {
    var: Arc<Variable>,
    selections: Arc<Vec<(Expr, Expr, Expr)>>,
}

impl Slice {
    pub fn make<T: IntoVar, U, A, B, C>(var: T, selections: U) -> Self
        where A: Into<Expr>, B: Into<Expr>, C: Into<Expr>, U: IntoIterator<Item = (A, B, C)>
    {
        Self {
            var: var.into_var().into(),
            selections: Arc::new(
                selections
                    .into_iter()
                    .map(|x| (x.0.into(), x.1.into(), x.2.into()))
                    .collect()
            ),
        }
    }
    pub fn var(&self) -> &Variable {
        &self.var
    }
    pub fn selections(&self) -> &[(Expr, Expr, Expr)] {
        &self.selections
    }
}

impl Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]",
            self.var,
            self.selections
                .iter()
                .map(|(start, end, step)| format!("{}:{}:{}", start, end, step))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct CommonReduce {
    value: Arc<Expr>,
    axes: Arc<Vec<Expr>>,
    closure: Closures,
}

impl CommonReduce {
    pub fn make<T: Into<Expr>, U: IntoIterator<Item: Into<Expr>>>(
        value: T,
        axes: U,
        closure: Closures
    ) -> Self {
        Self {
            value: value.into().into(),
            axes: axes
                .into_iter()
                .map(|x| x.into().into())
                .collect::<Vec<Expr>>()
                .into(),
            closure,
        }
    }
    pub fn value(&self) -> &Expr {
        &self.value
    }
    pub fn axes(&self) -> &[Expr] {
        &self.axes
    }
    pub fn closure(&self) -> &Closures {
        &self.closure
    }
}

impl Display for CommonReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]",
            self.value,
            self.axes
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

macro_rules! impl_into_expr {
    ($struct:ident) => {
        impl Into<Expr> for $struct {
            fn into(self) -> Expr {
                Expr::$struct(self)
            }
        }
        
        impl Into<Expr> for &$struct {
            fn into(self) -> Expr {
                Expr::$struct(self.clone())
            }
        }
    };
}

impl_into_expr!(Value);
impl_into_expr!(Str);
impl_into_expr!(Variable);
impl_into_expr!(Cast);
impl_into_expr!(Add);
impl_into_expr!(Sub);
impl_into_expr!(Mul);
impl_into_expr!(Div);
impl_into_expr!(Mod);
impl_into_expr!(Min);
impl_into_expr!(Max);
impl_into_expr!(Eq);
impl_into_expr!(Ne);
impl_into_expr!(Lt);
impl_into_expr!(Le);
impl_into_expr!(Gt);
impl_into_expr!(Ge);
impl_into_expr!(And);
impl_into_expr!(Or);
impl_into_expr!(Xor);
impl_into_expr!(Not);
impl_into_expr!(Call);
impl_into_expr!(Select);
impl_into_expr!(Let);
impl_into_expr!(Tensor);
impl_into_expr!(Alloc);
impl_into_expr!(If);
impl_into_expr!(For);
impl_into_expr!(While);
impl_into_expr!(Function);
impl_into_expr!(Tuple);
impl_into_expr!(TensorType);
impl_into_expr!(Slice);
impl_into_expr!(OpNode);
impl_into_expr!(ComputeNode);
impl_into_expr!(CommonReduce);

impl_binop!(Add, visit_add);
impl_binop!(Sub, visit_sub);
impl_binop!(Mul, visit_mul);
impl_binop!(Div, visit_div);
impl_binop!(Mod, visit_mod);
impl_binop!(Min, visit_min);
impl_binop!(Max, visit_max);
impl_binop!(Eq, visit_eq);
impl_binop!(Ne, visit_ne);
impl_binop!(Lt, visit_lt);
impl_binop!(Le, visit_le);
impl_binop!(Gt, visit_gt);
impl_binop!(Ge, visit_ge);
impl_binop!(And, visit_and);
impl_binop!(Or, visit_or);
impl_binop!(Xor, visit_xor);

impl IntoVar for Variable {
    fn into_var(self) -> Variable {
        self
    }
}

impl IntoVar for &Variable {
    fn into_var(self) -> Variable {
        self.clone()
    }
}

impl IntoVar for &[Variable] {
    fn into_var(self) -> Variable {
        Variable::make("")
    }
}

impl IntoVar for Arc<Variable> {
    fn into_var(self) -> Variable {
        self.as_ref().clone()
    }
}

impl IntoVar for &Arc<Variable> {
    fn into_var(self) -> Variable {
        self.as_ref().clone()
    }
}

impl IntoVar for &str {
    fn into_var(self) -> Variable {
        Variable::make(self)
    }
}

impl IntoVar for String {
    fn into_var(self) -> Variable {
        Variable::make(&self)
    }
}

impl IntoVar for &String {
    fn into_var(self) -> Variable {
        Variable::make(self)
    }
}
