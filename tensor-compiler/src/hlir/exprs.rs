use std::{ fmt::Display, sync::Arc };

use tensor_common::layout::Layout;
use tensor_types::dtype::Dtype;

use super::{ _value::_Value, node::Expr };

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

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
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
}

impl Display for Str {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone, PartialEq, Hash, Eq, Debug)]
pub struct Variable {
    value: Arc<String>,
}

impl Variable {
    pub fn make(value: &str) -> Self {
        Self {
            value: Arc::new(value.into()),
        }
    }
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.value)
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
}

impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({} as {})", self.expr, self.dtype)
    }
}

macro_rules! impl_binop {
    ($struct:ident) => {
        impl $struct {
            pub fn make<T: Into<Expr>, U: Into<Expr>>(lhs: T, rhs: U) -> Self {
                Self { lhs: lhs.into().into(), rhs: rhs.into().into() }
            }
        }
    };
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

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "!{}", self.expr)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Call {
    name: Arc<String>,
    args: Vec<Arc<Expr>>,
}

impl Call {
    pub fn make<T: Into<String>, U: IntoIterator<Item: Into<Expr>>>(name: T, args: U) -> Self {
        Self {
            name: Arc::new(name.into()),
            args: args
                .into_iter()
                .map(|x| x.into().into())
                .collect(),
        }
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
}

impl Let {
    pub fn make<T: Into<Variable>, U: Into<Expr>>(var: T, value: U) -> Self {
        Self {
            var: var.into().into(),
            value: value.into().into(),
        }
    }
}

impl Display for Let {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "let {} = {}", self.var, self.value)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Tensor {
    name: Arc<String>,
    layout: Arc<Layout>,
    dtype: Dtype,
}

impl Tensor {
    pub fn make<T: Into<String>>(name: T, layout: Layout, dtype: Dtype) -> Self {
        Self {
            name: Arc::new(name.into()),
            layout: Arc::new(layout),
            dtype,
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}{{{:?},{}}}", self.name, self.layout.shape().inner(), self.dtype)
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
    pub fn make<T: Into<Variable>, U: Into<Expr>, V: Into<Expr>, W: Into<Expr>, X: Into<Expr>>(
        var: T,
        start: U,
        end: V,
        step: W,
        body: X
    ) -> Self {
        Self {
            var: var.into().into(),
            start: start.into().into(),
            end: end.into().into(),
            step: step.into().into(),
            body: body.into().into(),
        }
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
}

impl Display for While {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "while {} {{\n{}\n}}", self.cond, self.body)
    }
}

#[derive(Clone, PartialEq, Debug, Hash, Eq)]
pub struct Function {
    name: Arc<String>,
    args: Vec<Arc<Variable>>,
    return_type: Arc<Expr>,
    body: Arc<Expr>,
}

impl Function {
    pub fn make<
        T: Into<String>,
        U: IntoIterator<Item: Into<Variable>>,
        V: Into<Expr>,
        W: Into<Expr>
    >(name: T, args: U, return_type: V, body: W) -> Self {
        Self {
            name: Arc::new(name.into()),
            args: args
                .into_iter()
                .map(|x| x.into().into())
                .collect(),
            return_type: return_type.into().into(),
            body: body.into().into(),
        }
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

impl_binop!(Add);
impl_binop!(Sub);
impl_binop!(Mul);
impl_binop!(Div);
impl_binop!(Mod);
impl_binop!(Min);
impl_binop!(Max);
impl_binop!(Eq);
impl_binop!(Ne);
impl_binop!(Lt);
impl_binop!(Le);
impl_binop!(Gt);
impl_binop!(Ge);
impl_binop!(And);
impl_binop!(Or);
impl_binop!(Xor);
